use std::io;

fn get_float_from_user(message: &str) -> f64 {
    loop {
        println!("{}", message);
        let mut input = String::new();

        // io::stdout().lock().write(message.as_ref()).unwrap();
        io::stdin()
            .read_line(&mut input)
            .expect("Not a valid string");
        println!("");
        let num = input.trim().parse();
        if num.is_ok() {
            return num.unwrap();
        }
    }
}

fn get_user_bounds() -> (f64, f64) {
    let user_bound_lower: f64 = get_float_from_user("Lower Bound: ");

    let user_bound_upper: f64 = get_float_from_user("Upper_bound: ");
    return (user_bound_lower, user_bound_upper);
}
fn ask_user_for_view(lower_bound: Option<f64>, upper_bound: Option<f64>) -> (f64, f64) {
    println!("Failed to determine boundary of the graph automatically.");
    println!("Pleas enter values manualy.");
    lower_bound.map(|b| println!("(Suggestion for lower bound: {})", b));
    upper_bound.map(|b| println!("(Suggestion for upper bound: {})", b));

    return get_user_bounds();
}
