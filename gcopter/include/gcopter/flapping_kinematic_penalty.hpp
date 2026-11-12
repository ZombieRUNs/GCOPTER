#ifndef GCOPTER_FLAPPING_KINEMATIC_PENALTY_HPP
#define GCOPTER_FLAPPING_KINEMATIC_PENALTY_HPP

#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gcopter
{
// Native whole-trajectory callback for GCOPTER_PolytopeSFC_Flapping.
// Coefficients are ascending powers, six rows per MINCO piece. The duration
// gradient is at fixed coefficients and includes both moving nodes and dt.
struct FlappingKinematicPenalty
{
    double velocity, acceleration, weight, epsilon;
    int intervals;

    FlappingKinematicPenalty(double v, double a, double w, int n, double e)
        : velocity(v), acceleration(a), weight(w), epsilon(e), intervals(n)
    {
        if (!std::isfinite(v) || !std::isfinite(a) || !std::isfinite(w) ||
            !std::isfinite(e) || v <= 0 || a <= 0 || w <= 0 || e <= 0 || n < 1)
            throw std::invalid_argument("positive finite flapping kinematic limits required");
    }

    double operator()(const Eigen::VectorXd &times, const Eigen::MatrixX3d &coeffs,
                      Eigen::VectorXd &gradT, Eigen::MatrixX3d &gradC,
                      double &violation) const
    {
        if (times.size() < 1 || coeffs.rows() != 6 * times.size() ||
            !times.allFinite() || !coeffs.allFinite() || times.minCoeff() <= 0)
            throw std::invalid_argument("invalid MINCO times or coefficients");
        gradT = Eigen::VectorXd::Zero(times.size());
        gradC = Eigen::MatrixX3d::Zero(coeffs.rows(), 3);
        double cost = 0.;
        violation = 0.;
        for (int i = 0; i < times.size(); ++i)
        {
            const auto c = coeffs.block<6,3>(6*i, 0);
            const double step = times(i) / intervals;
            for (int j = 0; j <= intervals; ++j)
            {
                const double alpha = double(j) / intervals, t = alpha * times(i);
                Eigen::Matrix<double,6,1> b1, b2, b3;
                b1 << 0.,1.,2*t,3*t*t,4*t*t*t,5*t*t*t*t;
                b2 << 0.,0.,2.,6*t,12*t*t,20*t*t*t;
                b3 << 0.,0.,0.,6.,24*t,60*t*t;
                const Eigen::Vector3d v = c.transpose()*b1;
                const Eigen::Vector3d a = c.transpose()*b2;
                const Eigen::Vector3d jerk = c.transpose()*b3;
                const double rv = v.squaredNorm()/(velocity*velocity)-1.;
                const double ra = a.squaredNorm()/(acceleration*acceleration)-1.;
                violation = std::max(violation, std::max(rv,ra));
                double pv,dv,pa,da;
                smooth(rv,pv,dv); smooth(ra,pa,da);
                const double penalty = weight*(pv+pa);
                const Eigen::Vector3d gv = (2*weight*dv/(velocity*velocity))*v;
                const Eigen::Vector3d ga = (2*weight*da/(acceleration*acceleration))*a;
                const double node = (j == 0 || j == intervals) ? .5 : 1.;
                cost += node*step*penalty;
                gradC.block<6,3>(6*i,0) += node*step*(b1*gv.transpose()+b2*ga.transpose());
                gradT(i) += node*(step*alpha*(gv.dot(a)+ga.dot(jerk))+penalty/intervals);
            }
        }
        return cost;
    }

private:
    void smooth(double x, double &value, double &derivative) const
    {
        if (x <= 0.) { value=derivative=0.; }
        else if (x >= epsilon) { value=x-.5*epsilon; derivative=1.; }
        else
        {
            const double r=x/epsilon;
            value=epsilon*(r*r*r-.5*r*r*r*r);
            derivative=3*r*r-2*r*r*r;
        }
    }
};
}
#endif
