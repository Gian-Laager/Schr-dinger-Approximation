use crate::*;
use std::fmt;

pub fn to_gnuplot_string_complex<X>(values: Vec<Point<X, Complex64>>) -> String
where
    X: fmt::Display + Send + Sync,
{
    values
        .par_iter()
        .map(|p| -> String { format!("{} {} {}\n", p.x, p.y.re, p.y.im) })
        .reduce(|| String::new(), |s: String, current: String| s + &*current)
}

pub fn to_gnuplot_string<X, Y>(values: Vec<Point<X, Y>>) -> String
where
    X: fmt::Display + Send + Sync,
    Y: fmt::Display + Send + Sync,
{
    values
        .par_iter()
        .map(|p| -> String { format!("{} {}\n", p.x, p.y) })
        .reduce(|| String::new(), |s: String, current: String| s + &*current)
}

pub fn plot_wavefunction_parts(wave_function: &WaveFunction, output_dir: &Path, output_file: &str) {
    std::env::set_current_dir(&output_dir).unwrap();

    let wkb_values = wave_function
        .get_wkb_ranges_in_view()
        .iter()
        .map(|range| evaluate_function_between(wave_function, range.0, range.1, NUMBER_OF_POINTS))
        .collect::<Vec<Vec<Point<f64, Complex64>>>>();

    let airy_values = wave_function
        .get_airy_ranges()
        .iter()
        .map(|range| {
            evaluate_function_between(
                wave_function,
                f64::max(wave_function.get_view().0, range.0),
                f64::min(wave_function.get_view().1, range.1),
                NUMBER_OF_POINTS,
            )
        })
        .collect::<Vec<Vec<Point<f64, Complex64>>>>();

    let wkb_values_str = wkb_values
        .par_iter()
        .map(|values| to_gnuplot_string_complex(values.to_vec()))
        .reduce(
            || String::new(),
            |s: String, current: String| s + "\n\n" + &*current,
        );

    let airy_values_str = airy_values
        .par_iter()
        .map(|values| to_gnuplot_string_complex(values.to_vec()))
        .reduce(
            || String::new(),
            |s: String, current: String| s + "\n\n" + &*current,
        );

    let mut data_full = File::create(output_file).unwrap();
    data_full.write_all(wkb_values_str.as_ref()).unwrap();
    data_full.write_all("\n\n".as_bytes()).unwrap();
    data_full.write_all(airy_values_str.as_ref()).unwrap();

    let mut plot_3d_file = File::create("plot_3d.gnuplot").unwrap();

    let wkb_3d_cmd = (1..=wkb_values.len())
        .into_iter()
        .map(|n| {
            format!(
                "\"{}\" u 1:2:3 i {} t \"WKB {}\" w l",
                output_file,
                n - 1,
                n
            )
        })
        .collect::<Vec<String>>()
        .join(", ");

    let airy_3d_cmd = (1..=airy_values.len())
        .into_iter()
        .map(|n| {
            format!(
                "\"{}\" u 1:2:3 i {} t \"Airy {}\" w l",
                output_file,
                n + wkb_values.len() - 1,
                n
            )
        })
        .collect::<Vec<String>>()
        .join(", ");
    let plot_3d_cmd: String = "splot ".to_string() + &wkb_3d_cmd + ", " + &airy_3d_cmd;
    plot_3d_file.write_all(plot_3d_cmd.as_ref()).unwrap();

    let mut plot_file = File::create("plot.gnuplot").unwrap();
    let wkb_cmd = (1..=wkb_values.len())
        .into_iter()
        .map(|n| {
            format!(
                "\"{}\" u 1:2 i {} t \"Re(WKB {})\" w l",
                output_file,
                n - 1,
                n
            )
        })
        .collect::<Vec<String>>()
        .join(", ");

    let airy_cmd = (1..=airy_values.len())
        .into_iter()
        .map(|n| {
            format!(
                "\"{}\" u 1:2 i {} t \"Re(Airy {})\" w l",
                output_file,
                n + wkb_values.len() - 1,
                n
            )
        })
        .collect::<Vec<String>>()
        .join(", ");
    let plot_cmd: String = "plot ".to_string() + &wkb_cmd + ", " + &airy_cmd;

    plot_file.write_all(plot_cmd.as_ref()).unwrap();

    let mut plot_imag_file = File::create("plot_im.gnuplot").unwrap();

    let wkb_im_cmd = (1..=wkb_values.len())
        .into_iter()
        .map(|n| {
            format!(
                "\"{}\" u 1:3 i {} t \"Im(WKB {})\" w l",
                output_file,
                n - 1,
                n
            )
        })
        .collect::<Vec<String>>()
        .join(", ");

    let airy_im_cmd = (1..=airy_values.len())
        .into_iter()
        .map(|n| {
            format!(
                "\"{}\" u 1:3 i {} t \"Im(Airy {})\" w l",
                output_file,
                n + wkb_values.len() - 1,
                n
            )
        })
        .collect::<Vec<String>>()
        .join(", ");
    let plot_imag_cmd: String = "plot ".to_string() + &wkb_im_cmd + ", " + &airy_im_cmd;

    plot_imag_file.write_all(plot_imag_cmd.as_ref()).unwrap();
}

pub fn plot_complex_function(
    func: &dyn Func<f64, Complex64>,
    view: (f64, f64),
    title: &str,
    output_dir: &Path,
    output_file: &str,
) {
    std::env::set_current_dir(&output_dir).unwrap();
    let values = evaluate_function_between(func, view.0, view.1, NUMBER_OF_POINTS);

    let values_str = to_gnuplot_string_complex(values);

    let mut data_file = File::create(output_file).unwrap();

    data_file.write_all(values_str.as_bytes()).unwrap();

    let mut plot_3d_file = File::create("plot_3d.gnuplot").unwrap();
    plot_3d_file
        .write_all(format!("splot \"{}\" u 1:2:3 t \"{}\" w l", output_file, title).as_bytes())
        .unwrap();

    let mut plot_file = File::create("plot.gnuplot").unwrap();
    plot_file
        .write_all(format!("plot \"{}\" u 1:2 t \"Re({})\" w l", output_file, title).as_bytes())
        .unwrap();

    let mut plot_im_file = File::create("plot_im.gnuplot").unwrap();
    plot_im_file
        .write_all(format!("plot \"{}\" u 1:3 t \"Im({})\" w l", output_file, title).as_bytes())
        .unwrap();
}

pub fn plot_wavefunction(wave_function: &WaveFunction, output_dir: &Path, output_file: &str) {
    plot_complex_function(
        wave_function,
        wave_function.get_view(),
        "Psi",
        output_dir,
        output_file,
    );
}

pub fn plot_superposition(wave_function: &SuperPosition, output_dir: &Path, output_file: &str) {
    plot_complex_function(
        wave_function,
        wave_function.get_view(),
        "Psi",
        output_dir,
        output_file,
    );
}

pub fn plot_probability(wave_function: &WaveFunction, output_dir: &Path, output_file: &str) {
    std::env::set_current_dir(&output_dir).unwrap();
    let values = evaluate_function_between(
        wave_function,
        wave_function.get_view().0,
        wave_function.get_view().1,
        NUMBER_OF_POINTS,
    )
    .par_iter()
    .map(|p| Point {
        x: p.x,
        y: p.y.norm_sqr(),
    })
    .collect();

    let values_str = to_gnuplot_string(values);

    let mut data_file = File::create(output_file).unwrap();

    data_file.write_all(values_str.as_bytes()).unwrap();

    let mut plot_file = File::create("plot.gnuplot").unwrap();
    plot_file
        .write_all(format!("plot \"{}\" u 1:2 t \"|Psi|^2\" w l", output_file).as_bytes())
        .unwrap();
}

pub fn plot_probability_super_pos(
    wave_function: &SuperPosition,
    output_dir: &Path,
    output_file: &str,
) {
    std::env::set_current_dir(&output_dir).unwrap();
    let values = evaluate_function_between(
        wave_function,
        wave_function.get_view().0,
        wave_function.get_view().1,
        NUMBER_OF_POINTS,
    ).par_iter()
    .map(|p| Point {
        x: p.x,
        y: p.y.norm_sqr(),
    })
    .collect();

    let values_str = to_gnuplot_string(values);

    let mut data_file = File::create(output_file).unwrap();

    data_file.write_all(values_str.as_bytes()).unwrap();

    let mut plot_file = File::create("plot.gnuplot").unwrap();
    plot_file
        .write_all(format!("plot \"{}\" u 1:2 t \"|Psi|^2\" w l", output_file).as_bytes())
        .unwrap();
}
