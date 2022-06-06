#pragma once

#include "pch.h"
#include "integration.h"

// fourier transform of V(x) as sum from 0 to n of a[n]*cos(b[n] * x)
class Potential
{
public:
    virtual double operator()(double x) = 0;
};

class StepPotential : public Potential, public Integrateable
{
public:
    struct Edge
    {
        double x;
        double yAfter;
    };
    double yStart;
    std::vector<Edge> edges;

    double operator()(double x) override
    {
        auto bigger = std::find_if(edges.begin(), edges.end(), [x](const Edge& e) { return e.x > x; });
        if (bigger == edges.begin())
            return yStart;
        auto edge = *(bigger - 1);
        return edge.yAfter;
    }

    double integral(double a, double b) override
    {
        if (a > b)
            std::swap(a, b);

        auto bigger = std::find_if(edges.begin(), edges.end(), [a](const Edge& e) { return e.x > a; });
        auto end = std::find_if(edges.rbegin(), edges.rend(), [b](const Edge& e) { return e.x < b; });
        if (bigger == edges.begin() && end == edges.rend())
            return (b - a) * yStart;
        else if (bigger == end.base())
            return (bigger - 1)->yAfter * (b - a);
        else if (bigger == edges.begin())
        {
            double integral = yStart * (edges[0].x - a);
            for (auto it = bigger; it != end.base() - 1; it++)
            {
                integral += ((it + 1)->x - it->x) * it->yAfter;
            }
            integral += (b - end->x) * end->yAfter;
            return integral;
        } else
        {
            double integral = 0;
            for (auto it = (bigger - 1); it != end.base() - 1; it++)
            {
                integral += ((it + 1)->x - it->x) * it->yAfter;
            }
            integral += (b - end->x) * end->yAfter;
            return integral;
        }
    }
};
