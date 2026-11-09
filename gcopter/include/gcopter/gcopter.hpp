/*
    MIT License

    Copyright (c) 2021 Zhepei Wang (wangzhepei@live.com)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
*/

#ifndef GCOPTER_HPP
#define GCOPTER_HPP

// geo_utils.hpp is required by processCorridor() below; it used to be
// pulled in implicitly by whatever the caller included first, which made
// gcopter.hpp unusable on its own.
#include "gcopter/geo_utils.hpp"

#include "gcopter/minco.hpp"
#include "gcopter/flatness.hpp"
#include "gcopter/flapping_flatness.hpp"
#include "gcopter/lbfgs.hpp"
#include "gcopter/trajectory.hpp"

#include <Eigen/Eigen>

#include <cmath>
#include <cfloat>
#include <iostream>
#include <limits>
#include <sstream>
#include <functional>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>

namespace gcopter
{

    class GCOPTER_PolytopeSFC
    {
    public:
        typedef Eigen::Matrix3Xd PolyhedronV;
        typedef Eigen::MatrixX4d PolyhedronH;
        typedef std::vector<PolyhedronV> PolyhedraV;
        typedef std::vector<PolyhedronH> PolyhedraH;

    private:
        minco::MINCO_S3NU minco;
        flatness::FlatnessMap flatmap;

        double rho;
        Eigen::Matrix3d headPVA;
        Eigen::Matrix3d tailPVA;

        PolyhedraV vPolytopes;
        PolyhedraH hPolytopes;
        Eigen::Matrix3Xd shortPath;

        Eigen::VectorXi pieceIdx;
        Eigen::VectorXi vPolyIdx;
        Eigen::VectorXi hPolyIdx;

        int polyN;
        int pieceN;

        int spatialDim;
        int temporalDim;

        double smoothEps;
        int integralRes;
        Eigen::VectorXd magnitudeBd;
        Eigen::VectorXd penaltyWt;
        Eigen::VectorXd physicalPm;
        double allocSpeed;

        lbfgs::lbfgs_parameter_t lbfgs_params;

        Eigen::Matrix3Xd points;
        Eigen::VectorXd times;
        Eigen::Matrix3Xd gradByPoints;
        Eigen::VectorXd gradByTimes;
        Eigen::MatrixX3d partialGradByCoeffs;
        Eigen::VectorXd partialGradByTimes;

    private:
        static inline void forwardT(const Eigen::VectorXd &tau,
                                    Eigen::VectorXd &T)
        {
            const int sizeTau = tau.size();
            T.resize(sizeTau);
            for (int i = 0; i < sizeTau; i++)
            {
                T(i) = tau(i) > 0.0
                           ? ((0.5 * tau(i) + 1.0) * tau(i) + 1.0)
                           : 1.0 / ((0.5 * tau(i) - 1.0) * tau(i) + 1.0);
            }
            return;
        }

        template <typename EIGENVEC>
        static inline void backwardT(const Eigen::VectorXd &T,
                                     EIGENVEC &tau)
        {
            const int sizeT = T.size();
            tau.resize(sizeT);
            for (int i = 0; i < sizeT; i++)
            {
                tau(i) = T(i) > 1.0
                             ? (sqrt(2.0 * T(i) - 1.0) - 1.0)
                             : (1.0 - sqrt(2.0 / T(i) - 1.0));
            }

            return;
        }

        template <typename EIGENVEC>
        static inline void backwardGradT(const Eigen::VectorXd &tau,
                                         const Eigen::VectorXd &gradT,
                                         EIGENVEC &gradTau)
        {
            const int sizeTau = tau.size();
            gradTau.resize(sizeTau);
            double denSqrt;
            for (int i = 0; i < sizeTau; i++)
            {
                if (tau(i) > 0)
                {
                    gradTau(i) = gradT(i) * (tau(i) + 1.0);
                }
                else
                {
                    denSqrt = (0.5 * tau(i) - 1.0) * tau(i) + 1.0;
                    gradTau(i) = gradT(i) * (1.0 - tau(i)) / (denSqrt * denSqrt);
                }
            }

            return;
        }

        static inline void forwardP(const Eigen::VectorXd &xi,
                                    const Eigen::VectorXi &vIdx,
                                    const PolyhedraV &vPolys,
                                    Eigen::Matrix3Xd &P)
        {
            const int sizeP = vIdx.size();
            P.resize(3, sizeP);
            Eigen::VectorXd q;
            for (int i = 0, j = 0, k, l; i < sizeP; i++, j += k)
            {
                l = vIdx(i);
                k = vPolys[l].cols();
                q = xi.segment(j, k).normalized().head(k - 1);
                P.col(i) = vPolys[l].rightCols(k - 1) * q.cwiseProduct(q) +
                           vPolys[l].col(0);
            }
            return;
        }

        static inline double costTinyNLS(void *ptr,
                                         const Eigen::VectorXd &xi,
                                         Eigen::VectorXd &gradXi)
        {
            const int n = xi.size();
            const Eigen::Matrix3Xd &ovPoly = *(Eigen::Matrix3Xd *)ptr;

            const double sqrNormXi = xi.squaredNorm();
            const double invNormXi = 1.0 / sqrt(sqrNormXi);
            const Eigen::VectorXd unitXi = xi * invNormXi;
            const Eigen::VectorXd r = unitXi.head(n - 1);
            const Eigen::Vector3d delta = ovPoly.rightCols(n - 1) * r.cwiseProduct(r) +
                                          ovPoly.col(1) - ovPoly.col(0);

            double cost = delta.squaredNorm();
            gradXi.head(n - 1) = (ovPoly.rightCols(n - 1).transpose() * (2 * delta)).array() *
                                 r.array() * 2.0;
            gradXi(n - 1) = 0.0;
            gradXi = (gradXi - unitXi.dot(gradXi) * unitXi).eval() * invNormXi;

            const double sqrNormViolation = sqrNormXi - 1.0;
            if (sqrNormViolation > 0.0)
            {
                double c = sqrNormViolation * sqrNormViolation;
                const double dc = 3.0 * c;
                c *= sqrNormViolation;
                cost += c;
                gradXi += dc * 2.0 * xi;
            }

            return cost;
        }

        template <typename EIGENVEC>
        static inline void backwardP(const Eigen::Matrix3Xd &P,
                                     const Eigen::VectorXi &vIdx,
                                     const PolyhedraV &vPolys,
                                     EIGENVEC &xi)
        {
            const int sizeP = P.cols();

            double minSqrD;
            lbfgs::lbfgs_parameter_t tiny_nls_params;
            tiny_nls_params.past = 0;
            tiny_nls_params.delta = 1.0e-5;
            tiny_nls_params.g_epsilon = FLT_EPSILON;
            tiny_nls_params.max_iterations = 128;

            Eigen::Matrix3Xd ovPoly;
            for (int i = 0, j = 0, k, l; i < sizeP; i++, j += k)
            {
                l = vIdx(i);
                k = vPolys[l].cols();

                ovPoly.resize(3, k + 1);
                ovPoly.col(0) = P.col(i);
                ovPoly.rightCols(k) = vPolys[l];
                Eigen::VectorXd x(k);
                x.setConstant(sqrt(1.0 / k));
                lbfgs::lbfgs_optimize(x,
                                      minSqrD,
                                      &GCOPTER_PolytopeSFC::costTinyNLS,
                                      nullptr,
                                      nullptr,
                                      &ovPoly,
                                      tiny_nls_params);

                xi.segment(j, k) = x;
            }

            return;
        }

        template <typename EIGENVEC>
        static inline void backwardGradP(const Eigen::VectorXd &xi,
                                         const Eigen::VectorXi &vIdx,
                                         const PolyhedraV &vPolys,
                                         const Eigen::Matrix3Xd &gradP,
                                         EIGENVEC &gradXi)
        {
            const int sizeP = vIdx.size();
            gradXi.resize(xi.size());

            double normInv;
            Eigen::VectorXd q, gradQ, unitQ;
            for (int i = 0, j = 0, k, l; i < sizeP; i++, j += k)
            {
                l = vIdx(i);
                k = vPolys[l].cols();
                q = xi.segment(j, k);
                normInv = 1.0 / q.norm();
                unitQ = q * normInv;
                gradQ.resize(k);
                gradQ.head(k - 1) = (vPolys[l].rightCols(k - 1).transpose() * gradP.col(i)).array() *
                                    unitQ.head(k - 1).array() * 2.0;
                gradQ(k - 1) = 0.0;
                gradXi.segment(j, k) = (gradQ - unitQ * unitQ.dot(gradQ)) * normInv;
            }

            return;
        }

        template <typename EIGENVEC>
        static inline void normRetrictionLayer(const Eigen::VectorXd &xi,
                                               const Eigen::VectorXi &vIdx,
                                               const PolyhedraV &vPolys,
                                               double &cost,
                                               EIGENVEC &gradXi)
        {
            const int sizeP = vIdx.size();
            gradXi.resize(xi.size());

            double sqrNormQ, sqrNormViolation, c, dc;
            Eigen::VectorXd q;
            for (int i = 0, j = 0, k; i < sizeP; i++, j += k)
            {
                k = vPolys[vIdx(i)].cols();

                q = xi.segment(j, k);
                sqrNormQ = q.squaredNorm();
                sqrNormViolation = sqrNormQ - 1.0;
                if (sqrNormViolation > 0.0)
                {
                    c = sqrNormViolation * sqrNormViolation;
                    dc = 3.0 * c;
                    c *= sqrNormViolation;
                    cost += c;
                    gradXi.segment(j, k) += dc * 2.0 * q;
                }
            }

            return;
        }

        static inline bool smoothedL1(const double &x,
                                      const double &mu,
                                      double &f,
                                      double &df)
        {
            if (x < 0.0)
            {
                return false;
            }
            else if (x > mu)
            {
                f = x - 0.5 * mu;
                df = 1.0;
                return true;
            }
            else
            {
                const double xdmu = x / mu;
                const double sqrxdmu = xdmu * xdmu;
                const double mumxd2 = mu - 0.5 * x;
                f = mumxd2 * sqrxdmu * xdmu;
                df = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
                return true;
            }
        }

        // magnitudeBounds = [v_max, omg_max, theta_max, thrust_min, thrust_max]^T
        // penaltyWeights = [pos_weight, vel_weight, omg_weight, theta_weight, thrust_weight]^T
        // physicalParams = [vehicle_mass, gravitational_acceleration, horitonral_drag_coeff,
        //                   vertical_drag_coeff, parasitic_drag_coeff, speed_smooth_factor]^T
        static inline void attachPenaltyFunctional(const Eigen::VectorXd &T,
                                                   const Eigen::MatrixX3d &coeffs,
                                                   const Eigen::VectorXi &hIdx,
                                                   const PolyhedraH &hPolys,
                                                   const double &smoothFactor,
                                                   const int &integralResolution,
                                                   const Eigen::VectorXd &magnitudeBounds,
                                                   const Eigen::VectorXd &penaltyWeights,
                                                   flatness::FlatnessMap &flatMap,
                                                   double &cost,
                                                   Eigen::VectorXd &gradT,
                                                   Eigen::MatrixX3d &gradC)
        {
            const double velSqrMax = magnitudeBounds(0) * magnitudeBounds(0);
            const double omgSqrMax = magnitudeBounds(1) * magnitudeBounds(1);
            const double thetaMax = magnitudeBounds(2);
            const double thrustMean = 0.5 * (magnitudeBounds(3) + magnitudeBounds(4));
            const double thrustRadi = 0.5 * fabs(magnitudeBounds(4) - magnitudeBounds(3));
            const double thrustSqrRadi = thrustRadi * thrustRadi;

            const double weightPos = penaltyWeights(0);
            const double weightVel = penaltyWeights(1);
            const double weightOmg = penaltyWeights(2);
            const double weightTheta = penaltyWeights(3);
            const double weightThrust = penaltyWeights(4);

            Eigen::Vector3d pos, vel, acc, jer, sna;
            Eigen::Vector3d totalGradPos, totalGradVel, totalGradAcc, totalGradJer;
            double totalGradPsi, totalGradPsiD;
            double thr, cos_theta;
            Eigen::Vector4d quat;
            Eigen::Vector3d omg;
            double gradThr;
            Eigen::Vector4d gradQuat;
            Eigen::Vector3d gradPos, gradVel, gradOmg;

            double step, alpha;
            double s1, s2, s3, s4, s5;
            Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3, beta4;
            Eigen::Vector3d outerNormal;
            int K, L;
            double violaPos, violaVel, violaOmg, violaTheta, violaThrust;
            double violaPosPenaD, violaVelPenaD, violaOmgPenaD, violaThetaPenaD, violaThrustPenaD;
            double violaPosPena, violaVelPena, violaOmgPena, violaThetaPena, violaThrustPena;
            double node, pena;

            const int pieceNum = T.size();
            const double integralFrac = 1.0 / integralResolution;
            for (int i = 0; i < pieceNum; i++)
            {
                const Eigen::Matrix<double, 6, 3> &c = coeffs.block<6, 3>(i * 6, 0);
                step = T(i) * integralFrac;
                for (int j = 0; j <= integralResolution; j++)
                {
                    s1 = j * step;
                    s2 = s1 * s1;
                    s3 = s2 * s1;
                    s4 = s2 * s2;
                    s5 = s4 * s1;
                    beta0(0) = 1.0, beta0(1) = s1, beta0(2) = s2, beta0(3) = s3, beta0(4) = s4, beta0(5) = s5;
                    beta1(0) = 0.0, beta1(1) = 1.0, beta1(2) = 2.0 * s1, beta1(3) = 3.0 * s2, beta1(4) = 4.0 * s3, beta1(5) = 5.0 * s4;
                    beta2(0) = 0.0, beta2(1) = 0.0, beta2(2) = 2.0, beta2(3) = 6.0 * s1, beta2(4) = 12.0 * s2, beta2(5) = 20.0 * s3;
                    beta3(0) = 0.0, beta3(1) = 0.0, beta3(2) = 0.0, beta3(3) = 6.0, beta3(4) = 24.0 * s1, beta3(5) = 60.0 * s2;
                    beta4(0) = 0.0, beta4(1) = 0.0, beta4(2) = 0.0, beta4(3) = 0.0, beta4(4) = 24.0, beta4(5) = 120.0 * s1;
                    pos = c.transpose() * beta0;
                    vel = c.transpose() * beta1;
                    acc = c.transpose() * beta2;
                    jer = c.transpose() * beta3;
                    sna = c.transpose() * beta4;

                    flatMap.forward(vel, acc, jer, 0.0, 0.0, thr, quat, omg);

                    violaVel = vel.squaredNorm() - velSqrMax;
                    violaOmg = omg.squaredNorm() - omgSqrMax;
                    cos_theta = 1.0 - 2.0 * (quat(1) * quat(1) + quat(2) * quat(2));
                    violaTheta = acos(cos_theta) - thetaMax;
                    violaThrust = (thr - thrustMean) * (thr - thrustMean) - thrustSqrRadi;

                    gradThr = 0.0;
                    gradQuat.setZero();
                    gradPos.setZero(), gradVel.setZero(), gradOmg.setZero();
                    pena = 0.0;

                    L = hIdx(i);
                    K = hPolys[L].rows();
                    for (int k = 0; k < K; k++)
                    {
                        outerNormal = hPolys[L].block<1, 3>(k, 0);
                        violaPos = outerNormal.dot(pos) + hPolys[L](k, 3);
                        if (smoothedL1(violaPos, smoothFactor, violaPosPena, violaPosPenaD))
                        {
                            gradPos += weightPos * violaPosPenaD * outerNormal;
                            pena += weightPos * violaPosPena;
                        }
                    }

                    if (smoothedL1(violaVel, smoothFactor, violaVelPena, violaVelPenaD))
                    {
                        gradVel += weightVel * violaVelPenaD * 2.0 * vel;
                        pena += weightVel * violaVelPena;
                    }

                    if (smoothedL1(violaOmg, smoothFactor, violaOmgPena, violaOmgPenaD))
                    {
                        gradOmg += weightOmg * violaOmgPenaD * 2.0 * omg;
                        pena += weightOmg * violaOmgPena;
                    }

                    if (smoothedL1(violaTheta, smoothFactor, violaThetaPena, violaThetaPenaD))
                    {
                        gradQuat += weightTheta * violaThetaPenaD /
                                    sqrt(1.0 - cos_theta * cos_theta) * 4.0 *
                                    Eigen::Vector4d(0.0, quat(1), quat(2), 0.0);
                        pena += weightTheta * violaThetaPena;
                    }

                    if (smoothedL1(violaThrust, smoothFactor, violaThrustPena, violaThrustPenaD))
                    {
                        gradThr += weightThrust * violaThrustPenaD * 2.0 * (thr - thrustMean);
                        pena += weightThrust * violaThrustPena;
                    }

                    flatMap.backward(gradPos, gradVel, gradThr, gradQuat, gradOmg,
                                     totalGradPos, totalGradVel, totalGradAcc, totalGradJer,
                                     totalGradPsi, totalGradPsiD);

                    node = (j == 0 || j == integralResolution) ? 0.5 : 1.0;
                    alpha = j * integralFrac;
                    gradC.block<6, 3>(i * 6, 0) += (beta0 * totalGradPos.transpose() +
                                                    beta1 * totalGradVel.transpose() +
                                                    beta2 * totalGradAcc.transpose() +
                                                    beta3 * totalGradJer.transpose()) *
                                                   node * step;
                    gradT(i) += (totalGradPos.dot(vel) +
                                 totalGradVel.dot(acc) +
                                 totalGradAcc.dot(jer) +
                                 totalGradJer.dot(sna)) *
                                    alpha * node * step +
                                node * integralFrac * pena;
                    cost += node * step * pena;
                }
            }

            return;
        }

        static inline double costFunctional(void *ptr,
                                            const Eigen::VectorXd &x,
                                            Eigen::VectorXd &g)
        {
            GCOPTER_PolytopeSFC &obj = *(GCOPTER_PolytopeSFC *)ptr;
            const int dimTau = obj.temporalDim;
            const int dimXi = obj.spatialDim;
            const double weightT = obj.rho;
            Eigen::Map<const Eigen::VectorXd> tau(x.data(), dimTau);
            Eigen::Map<const Eigen::VectorXd> xi(x.data() + dimTau, dimXi);
            Eigen::Map<Eigen::VectorXd> gradTau(g.data(), dimTau);
            Eigen::Map<Eigen::VectorXd> gradXi(g.data() + dimTau, dimXi);

            forwardT(tau, obj.times);
            forwardP(xi, obj.vPolyIdx, obj.vPolytopes, obj.points);

            double cost;
            obj.minco.setParameters(obj.points, obj.times);
            obj.minco.getEnergy(cost);
            obj.minco.getEnergyPartialGradByCoeffs(obj.partialGradByCoeffs);
            obj.minco.getEnergyPartialGradByTimes(obj.partialGradByTimes);

            attachPenaltyFunctional(obj.times, obj.minco.getCoeffs(),
                                    obj.hPolyIdx, obj.hPolytopes,
                                    obj.smoothEps, obj.integralRes,
                                    obj.magnitudeBd, obj.penaltyWt, obj.flatmap,
                                    cost, obj.partialGradByTimes, obj.partialGradByCoeffs);

            obj.minco.propogateGrad(obj.partialGradByCoeffs, obj.partialGradByTimes,
                                    obj.gradByPoints, obj.gradByTimes);

            cost += weightT * obj.times.sum();
            obj.gradByTimes.array() += weightT;

            backwardGradT(tau, obj.gradByTimes, gradTau);
            backwardGradP(xi, obj.vPolyIdx, obj.vPolytopes, obj.gradByPoints, gradXi);
            normRetrictionLayer(xi, obj.vPolyIdx, obj.vPolytopes, cost, gradXi);

            return cost;
        }

        static inline double costDistance(void *ptr,
                                          const Eigen::VectorXd &xi,
                                          Eigen::VectorXd &gradXi)
        {
            void **dataPtrs = (void **)ptr;
            const double &dEps = *((const double *)(dataPtrs[0]));
            const Eigen::Vector3d &ini = *((const Eigen::Vector3d *)(dataPtrs[1]));
            const Eigen::Vector3d &fin = *((const Eigen::Vector3d *)(dataPtrs[2]));
            const PolyhedraV &vPolys = *((PolyhedraV *)(dataPtrs[3]));

            double cost = 0.0;
            const int overlaps = vPolys.size() / 2;

            Eigen::Matrix3Xd gradP = Eigen::Matrix3Xd::Zero(3, overlaps);
            Eigen::Vector3d a, b, d;
            Eigen::VectorXd r;
            double smoothedDistance;
            for (int i = 0, j = 0, k = 0; i <= overlaps; i++, j += k)
            {
                a = i == 0 ? ini : b;
                if (i < overlaps)
                {
                    k = vPolys[2 * i + 1].cols();
                    Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
                    r = q.normalized().head(k - 1);
                    b = vPolys[2 * i + 1].rightCols(k - 1) * r.cwiseProduct(r) +
                        vPolys[2 * i + 1].col(0);
                }
                else
                {
                    b = fin;
                }

                d = b - a;
                smoothedDistance = sqrt(d.squaredNorm() + dEps);
                cost += smoothedDistance;

                if (i < overlaps)
                {
                    gradP.col(i) += d / smoothedDistance;
                }
                if (i > 0)
                {
                    gradP.col(i - 1) -= d / smoothedDistance;
                }
            }

            Eigen::VectorXd unitQ;
            double sqrNormQ, invNormQ, sqrNormViolation, c, dc;
            for (int i = 0, j = 0, k; i < overlaps; i++, j += k)
            {
                k = vPolys[2 * i + 1].cols();
                Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
                Eigen::Map<Eigen::VectorXd> gradQ(gradXi.data() + j, k);
                sqrNormQ = q.squaredNorm();
                invNormQ = 1.0 / sqrt(sqrNormQ);
                unitQ = q * invNormQ;
                gradQ.head(k - 1) = (vPolys[2 * i + 1].rightCols(k - 1).transpose() * gradP.col(i)).array() *
                                    unitQ.head(k - 1).array() * 2.0;
                gradQ(k - 1) = 0.0;
                gradQ = (gradQ - unitQ * unitQ.dot(gradQ)).eval() * invNormQ;

                sqrNormViolation = sqrNormQ - 1.0;
                if (sqrNormViolation > 0.0)
                {
                    c = sqrNormViolation * sqrNormViolation;
                    dc = 3.0 * c;
                    c *= sqrNormViolation;
                    cost += c;
                    gradQ += dc * 2.0 * q;
                }
            }

            return cost;
        }

        static inline void getShortestPath(const Eigen::Vector3d &ini,
                                           const Eigen::Vector3d &fin,
                                           const PolyhedraV &vPolys,
                                           const double &smoothD,
                                           Eigen::Matrix3Xd &path)
        {
            const int overlaps = vPolys.size() / 2;
            Eigen::VectorXi vSizes(overlaps);
            for (int i = 0; i < overlaps; i++)
            {
                vSizes(i) = vPolys[2 * i + 1].cols();
            }
            Eigen::VectorXd xi(vSizes.sum());
            for (int i = 0, j = 0; i < overlaps; i++)
            {
                xi.segment(j, vSizes(i)).setConstant(sqrt(1.0 / vSizes(i)));
                j += vSizes(i);
            }

            double minDistance;
            void *dataPtrs[4];
            dataPtrs[0] = (void *)(&smoothD);
            dataPtrs[1] = (void *)(&ini);
            dataPtrs[2] = (void *)(&fin);
            dataPtrs[3] = (void *)(&vPolys);
            lbfgs::lbfgs_parameter_t shortest_path_params;
            shortest_path_params.past = 3;
            shortest_path_params.delta = 1.0e-3;
            shortest_path_params.g_epsilon = 1.0e-5;

            lbfgs::lbfgs_optimize(xi,
                                  minDistance,
                                  &GCOPTER_PolytopeSFC::costDistance,
                                  nullptr,
                                  nullptr,
                                  dataPtrs,
                                  shortest_path_params);

            path.resize(3, overlaps + 2);
            path.leftCols<1>() = ini;
            path.rightCols<1>() = fin;
            Eigen::VectorXd r;
            for (int i = 0, j = 0, k; i < overlaps; i++, j += k)
            {
                k = vPolys[2 * i + 1].cols();
                Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
                r = q.normalized().head(k - 1);
                path.col(i + 1) = vPolys[2 * i + 1].rightCols(k - 1) * r.cwiseProduct(r) +
                                  vPolys[2 * i + 1].col(0);
            }

            return;
        }

        static inline bool processCorridor(const PolyhedraH &hPs,
                                           PolyhedraV &vPs)
        {
            const int sizeCorridor = hPs.size() - 1;

            vPs.clear();
            vPs.reserve(2 * sizeCorridor + 1);

            int nv;
            PolyhedronH curIH;
            PolyhedronV curIV, curIOB;
            for (int i = 0; i < sizeCorridor; i++)
            {
                if (!geo_utils::enumerateVs(hPs[i], curIV))
                {
                    return false;
                }
                nv = curIV.cols();
                curIOB.resize(3, nv);
                curIOB.col(0) = curIV.col(0);
                curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
                vPs.push_back(curIOB);

                curIH.resize(hPs[i].rows() + hPs[i + 1].rows(), 4);
                curIH.topRows(hPs[i].rows()) = hPs[i];
                curIH.bottomRows(hPs[i + 1].rows()) = hPs[i + 1];
                if (!geo_utils::enumerateVs(curIH, curIV))
                {
                    return false;
                }
                nv = curIV.cols();
                curIOB.resize(3, nv);
                curIOB.col(0) = curIV.col(0);
                curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
                vPs.push_back(curIOB);
            }

            if (!geo_utils::enumerateVs(hPs.back(), curIV))
            {
                return false;
            }
            nv = curIV.cols();
            curIOB.resize(3, nv);
            curIOB.col(0) = curIV.col(0);
            curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
            vPs.push_back(curIOB);

            return true;
        }

        static inline void setInitial(const Eigen::Matrix3Xd &path,
                                      const double &speed,
                                      const Eigen::VectorXi &intervalNs,
                                      Eigen::Matrix3Xd &innerPoints,
                                      Eigen::VectorXd &timeAlloc)
        {
            const int sizeM = intervalNs.size();
            const int sizeN = intervalNs.sum();
            innerPoints.resize(3, sizeN - 1);
            timeAlloc.resize(sizeN);

            Eigen::Vector3d a, b, c;
            for (int i = 0, j = 0, k = 0, l; i < sizeM; i++)
            {
                l = intervalNs(i);
                a = path.col(i);
                b = path.col(i + 1);
                c = (b - a) / l;
                timeAlloc.segment(j, l).setConstant(c.norm() / speed);
                j += l;
                for (int m = 0; m < l; m++)
                {
                    if (i > 0 || m > 0)
                    {
                        innerPoints.col(k++) = a + c * m;
                    }
                }
            }
        }

    public:
        // magnitudeBounds = [v_max, omg_max, theta_max, thrust_min, thrust_max]^T
        // penaltyWeights = [pos_weight, vel_weight, omg_weight, theta_weight, thrust_weight]^T
        // physicalParams = [vehicle_mass, gravitational_acceleration, horitonral_drag_coeff,
        //                   vertical_drag_coeff, parasitic_drag_coeff, speed_smooth_factor]^T
        inline bool setup(const double &timeWeight,
                          const Eigen::Matrix3d &initialPVA,
                          const Eigen::Matrix3d &terminalPVA,
                          const PolyhedraH &safeCorridor,
                          const double &lengthPerPiece,
                          const double &smoothingFactor,
                          const int &integralResolution,
                          const Eigen::VectorXd &magnitudeBounds,
                          const Eigen::VectorXd &penaltyWeights,
                          const Eigen::VectorXd &physicalParams)
        {
            rho = timeWeight;
            headPVA = initialPVA;
            tailPVA = terminalPVA;

            hPolytopes = safeCorridor;
            for (size_t i = 0; i < hPolytopes.size(); i++)
            {
                const Eigen::ArrayXd norms =
                    hPolytopes[i].leftCols<3>().rowwise().norm();
                hPolytopes[i].array().colwise() /= norms;
            }
            if (!processCorridor(hPolytopes, vPolytopes))
            {
                return false;
            }

            polyN = hPolytopes.size();
            smoothEps = smoothingFactor;
            integralRes = integralResolution;
            magnitudeBd = magnitudeBounds;
            penaltyWt = penaltyWeights;
            physicalPm = physicalParams;
            allocSpeed = magnitudeBd(0) * 3.0;

            getShortestPath(headPVA.col(0), tailPVA.col(0),
                            vPolytopes, smoothEps, shortPath);
            const Eigen::Matrix3Xd deltas = shortPath.rightCols(polyN) - shortPath.leftCols(polyN);
            pieceIdx = (deltas.colwise().norm() / lengthPerPiece).cast<int>().transpose();
            pieceIdx.array() += 1;
            pieceN = pieceIdx.sum();

            temporalDim = pieceN;
            spatialDim = 0;
            vPolyIdx.resize(pieceN - 1);
            hPolyIdx.resize(pieceN);
            for (int i = 0, j = 0, k; i < polyN; i++)
            {
                k = pieceIdx(i);
                for (int l = 0; l < k; l++, j++)
                {
                    if (l < k - 1)
                    {
                        vPolyIdx(j) = 2 * i;
                        spatialDim += vPolytopes[2 * i].cols();
                    }
                    else if (i < polyN - 1)
                    {
                        vPolyIdx(j) = 2 * i + 1;
                        spatialDim += vPolytopes[2 * i + 1].cols();
                    }
                    hPolyIdx(j) = i;
                }
            }

            // Setup for MINCO_S3NU, FlatnessMap, and L-BFGS solver
            minco.setConditions(headPVA, tailPVA, pieceN);
            flatmap.reset(physicalPm(0), physicalPm(1), physicalPm(2),
                          physicalPm(3), physicalPm(4), physicalPm(5));

            // Allocate temp variables
            points.resize(3, pieceN - 1);
            times.resize(pieceN);
            gradByPoints.resize(3, pieceN - 1);
            gradByTimes.resize(pieceN);
            partialGradByCoeffs.resize(6 * pieceN, 3);
            partialGradByTimes.resize(pieceN);

            return true;
        }

        inline double optimize(Trajectory<5> &traj,
                               const double &relCostTol)
        {
            Eigen::VectorXd x(temporalDim + spatialDim);
            Eigen::Map<Eigen::VectorXd> tau(x.data(), temporalDim);
            Eigen::Map<Eigen::VectorXd> xi(x.data() + temporalDim, spatialDim);

            setInitial(shortPath, allocSpeed, pieceIdx, points, times);
            backwardT(times, tau);
            backwardP(points, vPolyIdx, vPolytopes, xi);

            double minCostFunctional;
            lbfgs_params.mem_size = 256;
            lbfgs_params.past = 3;
            lbfgs_params.min_step = 1.0e-32;
            lbfgs_params.g_epsilon = 0.0;
            lbfgs_params.delta = relCostTol;

            int ret = lbfgs::lbfgs_optimize(x,
                                            minCostFunctional,
                                            &GCOPTER_PolytopeSFC::costFunctional,
                                            nullptr,
                                            nullptr,
                                            this,
                                            lbfgs_params);

            if (ret >= 0)
            {
                forwardT(tau, times);
                forwardP(xi, vPolyIdx, vPolytopes, points);
                minco.setParameters(points, times);
                minco.getTrajectory(traj);
            }
            else
            {
                traj.clear();
                minCostFunctional = INFINITY;
                std::cout << "Optimization Failed: "
                          << lbfgs::lbfgs_strerror(ret)
                          << std::endl;
            }

            return minCostFunctional;
        }
    };

    // =======================================================================
    // Flapping-wing constraint configuration
    //
    // The quadrotor planner packs its limits into
    //     magnitudeBounds = [v_max, omg_max, theta_max, thrust_min, thrust_max]
    // and reaches them by position. The flapping model constrains a different,
    // longer list (per-axis rates and moments, an explicit thrust interval, an
    // optional tilt limit), so it gets its own vectors of its own length and
    // its own named indices below. Passing a quadrotor vector to this planner
    // fails the size check in setup() instead of silently reading the wrong
    // slot.
    // =======================================================================

    // flappingBounds: FLAP_BOUND_DIM = 11 entries, all in SI except the enable
    // flag.
    enum FlappingBoundIndex
    {
        FLAP_BOUND_VEL_MAX = 0,     // ||v|| upper bound [m/s], > 0
        FLAP_BOUND_THRUST_MIN = 1,  // T lower bound [N], >= 0
        FLAP_BOUND_THRUST_MAX = 2,  // T upper bound [N], > thrust_min
        FLAP_BOUND_OMEGA_MAX_X = 3, // |omega_x| upper bound [rad/s], > 0
        FLAP_BOUND_OMEGA_MAX_Y = 4, // |omega_y| upper bound [rad/s], > 0
        FLAP_BOUND_OMEGA_MAX_Z = 5, // |omega_z| upper bound [rad/s], > 0
        FLAP_BOUND_GAMMA_MAX_X = 6, // |Gamma_x| upper bound [N m], > 0
        FLAP_BOUND_GAMMA_MAX_Y = 7, // |Gamma_y| upper bound [N m], > 0
        FLAP_BOUND_GAMMA_MAX_Z = 8, // |Gamma_z| upper bound [N m], > 0
        FLAP_BOUND_TILT_MAX = 9,    // tilt angle upper bound [rad], in (0, pi/2]
        FLAP_BOUND_TILT_ENABLE = 10,// 1 to constrain tilt, 0 to leave it free
        FLAP_BOUND_DIM = 11
    };

    // flappingWeights: FLAP_WEIGHT_DIM = 7 entries, each >= 0. A zero weight
    // disables that term.
    enum FlappingWeightIndex
    {
        FLAP_WEIGHT_POSITION = 0,   // position-corridor violation
        FLAP_WEIGHT_VELOCITY = 1,   // ||v|| <= v_max
        FLAP_WEIGHT_THRUST = 2,     // T_min <= T <= T_max (both sides)
        FLAP_WEIGHT_OMEGA = 3,      // |omega_i| <= omega_max_i
        FLAP_WEIGHT_GAMMA = 4,      // |Gamma_i| <= Gamma_max_i
        FLAP_WEIGHT_TILT = 5,       // tilt <= tilt_max
        FLAP_WEIGHT_DEGENERATE = 6, // node where the flatness map has no attitude
        FLAP_WEIGHT_DIM = 7
    };

    // flappingPhysical: FLAP_PHYSICAL_DIM = 5 entries. This is exactly the
    // parameter set of the simplified model: m, J and g.
    enum FlappingPhysicalIndex
    {
        FLAP_PHYSICAL_MASS = 0,     // m [kg], > 0
        FLAP_PHYSICAL_GRAVITY = 1,  // g [m/s^2], >= 0
        FLAP_PHYSICAL_INERTIA_X = 2,// J_xx [kg m^2], > 0
        FLAP_PHYSICAL_INERTIA_Y = 3,// J_yy [kg m^2], > 0
        FLAP_PHYSICAL_INERTIA_Z = 4,// J_zz [kg m^2], > 0
        FLAP_PHYSICAL_DIM = 5
    };

    // flappingAllocation: FLAP_ALLOC_DIM = 3 entries. The left/right wing
    // thrust allocation the total-thrust / total-roll coupling comes from:
    //   T = T_L + T_R,  Gamma_x = l_eff (T_R - T_L),
    // which every node must satisfy with F_min <= T_L, T_R <= F_max. This is a
    // genuinely new constraint set, not a restatement of the total-thrust or
    // |Gamma_i| bounds: a node can meet both of those and still leave one side
    // outside [F_min, F_max].
    enum FlappingAllocationIndex
    {
        FLAP_ALLOC_ROLL_ARM = 0,   // l_eff [m], > 0 (single-side effective arm)
        FLAP_ALLOC_WING_MIN = 1,   // F_min [N], >= 0 (per-side lower bound)
        FLAP_ALLOC_WING_MAX = 2,   // F_max [N], > F_min (per-side upper bound)
        FLAP_ALLOC_DIM = 3
    };

    // Which constraint produced the worst violation, for reporting.
    enum class FlappingViolation
    {
        None = 0,
        Corridor = 1,
        Speed = 2,
        ThrustLow = 3,
        ThrustHigh = 4,
        Omega = 5,
        Gamma = 6,
        Tilt = 7,
        Degenerate = 8,
        // The four left/right wing-thrust bounds. T_L is the lower-thrust side
        // when Gamma_x > 0, T_R the higher one; the names follow the wing, not
        // the sign of Gamma_x.
        WingLeftLow = 9,
        WingLeftHigh = 10,
        WingRightLow = 11,
        WingRightHigh = 12,
        Additional = 13
    };

    inline const char *flappingViolationName(const FlappingViolation v)
    {
        switch (v)
        {
        case FlappingViolation::None:
            return "None";
        case FlappingViolation::Corridor:
            return "Corridor";
        case FlappingViolation::Speed:
            return "Speed";
        case FlappingViolation::ThrustLow:
            return "ThrustLow";
        case FlappingViolation::ThrustHigh:
            return "ThrustHigh";
        case FlappingViolation::Omega:
            return "Omega";
        case FlappingViolation::Gamma:
            return "Gamma";
        case FlappingViolation::Tilt:
            return "Tilt";
        case FlappingViolation::Degenerate:
            return "Degenerate";
        case FlappingViolation::WingLeftLow:
            return "WingLeftLow";
        case FlappingViolation::WingLeftHigh:
            return "WingLeftHigh";
        case FlappingViolation::WingRightLow:
            return "WingRightLow";
        case FlappingViolation::WingRightHigh:
            return "WingRightHigh";
        case FlappingViolation::Additional:
            return "Additional";
        }
        return "Unknown";
    }

    // Validated flapping configuration. parse() is the only way to fill one, so
    // an invalid vector can never reach the planner half-applied.
    struct FlappingConfig
    {
        double velMax = 0.0;
        double thrustMin = 0.0;
        double thrustMax = 0.0;
        Eigen::Vector3d omegaMax = Eigen::Vector3d::Zero();
        Eigen::Vector3d gammaMax = Eigen::Vector3d::Zero();
        double tiltMax = 0.0;
        bool tiltEnabled = false;

        double weightPosition = 0.0;
        double weightVelocity = 0.0;
        double weightThrust = 0.0;
        double weightOmega = 0.0;
        double weightGamma = 0.0;
        double weightTilt = 0.0;
        double weightDegenerate = 0.0;

        double mass = 0.0;
        double gravity = 0.0;
        Eigen::Vector3d inertia = Eigen::Vector3d::Zero();

        // Left/right wing thrust allocation. l_eff is the single-side roll arm
        // [m]; wingMin/wingMax are the per-side thrust bounds [N]. weightWing is
        // the scalar weight of the four per-side penalties.
        double rollArm = 0.0;
        double wingThrustMin = 0.0;
        double wingThrustMax = 0.0;
        double weightWingThrust = 0.0;

        // Length and value validation. Returns false and fills `reason` with a
        // message naming the offending entry; nothing is written to `cfg` in
        // that case.
        //
        // The allocation vector and the wing-weight scalar are explicit
        // arguments rather than fields of `bounds`/`weights`: the existing
        // 11/7/5 entry layouts above are unchanged, and the four per-side
        // bounds are a separate constraint set. There is deliberately no
        // compiled-in production default for them -- a caller has to supply
        // them, so a test or a stale config cannot silently inherit the shipped
        // vehicle's numbers.
        static inline bool parse(const Eigen::VectorXd &bounds,
                                 const Eigen::VectorXd &weights,
                                 const Eigen::VectorXd &physical,
                                 const Eigen::VectorXd &allocation,
                                 const double weightWingThrust,
                                 FlappingConfig &cfg,
                                 std::string &reason)
        {
            const auto fail = [&reason](const std::string &text) {
                reason = text;
                return false;
            };

            if (bounds.size() != FLAP_BOUND_DIM)
            {
                return fail("flappingBounds must have FLAP_BOUND_DIM = " +
                            std::to_string(FLAP_BOUND_DIM) + " entries, got " +
                            std::to_string(bounds.size()));
            }
            if (weights.size() != FLAP_WEIGHT_DIM)
            {
                return fail("flappingWeights must have FLAP_WEIGHT_DIM = " +
                            std::to_string(FLAP_WEIGHT_DIM) + " entries, got " +
                            std::to_string(weights.size()));
            }
            if (physical.size() != FLAP_PHYSICAL_DIM)
            {
                return fail("flappingPhysical must have FLAP_PHYSICAL_DIM = " +
                            std::to_string(FLAP_PHYSICAL_DIM) + " entries, got " +
                            std::to_string(physical.size()));
            }
            if (allocation.size() != FLAP_ALLOC_DIM)
            {
                return fail("flappingAllocation must have FLAP_ALLOC_DIM = " +
                            std::to_string(FLAP_ALLOC_DIM) + " entries, got " +
                            std::to_string(allocation.size()));
            }
            if (!flapping::detail::isFinite(bounds))
            {
                return fail("flappingBounds contains NaN or Inf");
            }
            if (!flapping::detail::isFinite(weights))
            {
                return fail("flappingWeights contains NaN or Inf");
            }
            if (!flapping::detail::isFinite(physical))
            {
                return fail("flappingPhysical contains NaN or Inf");
            }
            if (!flapping::detail::isFinite(allocation))
            {
                return fail("flappingAllocation contains NaN or Inf");
            }
            if (!flapping::detail::isFinite(weightWingThrust))
            {
                return fail("weightWingThrust contains NaN or Inf");
            }

            FlappingConfig parsed;
            const auto named = [](const char *field, const char *why) {
                return std::string(field) + " " + why;
            };

            parsed.velMax = bounds(FLAP_BOUND_VEL_MAX);
            if (!(parsed.velMax > 0.0))
            {
                return fail(named("flappingBounds[VEL_MAX]", "must be > 0"));
            }
            parsed.thrustMin = bounds(FLAP_BOUND_THRUST_MIN);
            parsed.thrustMax = bounds(FLAP_BOUND_THRUST_MAX);
            if (!(parsed.thrustMin >= 0.0))
            {
                return fail(named("flappingBounds[THRUST_MIN]", "must be >= 0"));
            }
            if (!(parsed.thrustMax > parsed.thrustMin))
            {
                return fail(named("flappingBounds[THRUST_MAX]",
                                  "must be > THRUST_MIN"));
            }
            for (int a = 0; a < 3; a++)
            {
                parsed.omegaMax(a) = bounds(FLAP_BOUND_OMEGA_MAX_X + a);
                if (!(parsed.omegaMax(a) > 0.0))
                {
                    return fail(named("flappingBounds[OMEGA_MAX_*]",
                                      "every axis bound must be > 0"));
                }
                parsed.gammaMax(a) = bounds(FLAP_BOUND_GAMMA_MAX_X + a);
                if (!(parsed.gammaMax(a) > 0.0))
                {
                    return fail(named("flappingBounds[GAMMA_MAX_*]",
                                      "every axis bound must be > 0"));
                }
            }
            parsed.tiltMax = bounds(FLAP_BOUND_TILT_MAX);
            if (!(parsed.tiltMax > 0.0) || !(parsed.tiltMax <= 0.5 * M_PI))
            {
                return fail(named("flappingBounds[TILT_MAX]",
                                  "must be in (0, pi/2]"));
            }
            const double tiltEnable = bounds(FLAP_BOUND_TILT_ENABLE);
            if (tiltEnable != 0.0 && tiltEnable != 1.0)
            {
                return fail(named("flappingBounds[TILT_ENABLE]",
                                  "must be exactly 0 or 1"));
            }
            parsed.tiltEnabled = tiltEnable == 1.0;

            parsed.weightPosition = weights(FLAP_WEIGHT_POSITION);
            parsed.weightVelocity = weights(FLAP_WEIGHT_VELOCITY);
            parsed.weightThrust = weights(FLAP_WEIGHT_THRUST);
            parsed.weightOmega = weights(FLAP_WEIGHT_OMEGA);
            parsed.weightGamma = weights(FLAP_WEIGHT_GAMMA);
            parsed.weightTilt = weights(FLAP_WEIGHT_TILT);
            parsed.weightDegenerate = weights(FLAP_WEIGHT_DEGENERATE);
            for (int k = 0; k < FLAP_WEIGHT_DIM; k++)
            {
                if (!(weights(k) >= 0.0))
                {
                    return fail(named("flappingWeights[*]", "must be >= 0"));
                }
            }

            parsed.mass = physical(FLAP_PHYSICAL_MASS);
            parsed.gravity = physical(FLAP_PHYSICAL_GRAVITY);
            parsed.inertia = Eigen::Vector3d(physical(FLAP_PHYSICAL_INERTIA_X),
                                             physical(FLAP_PHYSICAL_INERTIA_Y),
                                             physical(FLAP_PHYSICAL_INERTIA_Z));
            if (!(parsed.mass > 0.0))
            {
                return fail(named("flappingPhysical[MASS]", "must be > 0"));
            }
            if (!(parsed.gravity >= 0.0))
            {
                return fail(named("flappingPhysical[GRAVITY]", "must be >= 0"));
            }
            for (int a = 0; a < 3; a++)
            {
                if (!(parsed.inertia(a) > 0.0))
                {
                    return fail(named("flappingPhysical[INERTIA_*]",
                                      "every diagonal entry must be > 0"));
                }
            }

            parsed.rollArm = allocation(FLAP_ALLOC_ROLL_ARM);
            if (!(parsed.rollArm > 0.0))
            {
                return fail(named("flappingAllocation[ROLL_ARM]", "must be > 0"));
            }
            parsed.wingThrustMin = allocation(FLAP_ALLOC_WING_MIN);
            parsed.wingThrustMax = allocation(FLAP_ALLOC_WING_MAX);
            if (!(parsed.wingThrustMin >= 0.0))
            {
                return fail(named("flappingAllocation[WING_MIN]", "must be >= 0"));
            }
            if (!(parsed.wingThrustMax > parsed.wingThrustMin))
            {
                return fail(named("flappingAllocation[WING_MAX]",
                                  "must be > WING_MIN"));
            }
            if (!(weightWingThrust >= 0.0))
            {
                return fail(named("weightWingThrust", "must be >= 0"));
            }
            // The total-thrust interval has to sit inside the range the two
            // sides can actually produce. A total bound wider than
            // [2 F_min, 2 F_max] would let the optimiser command a total thrust
            // no admissible split can realise, which is exactly the coupling
            // this constraint set adds.
            if (parsed.thrustMin < 2.0 * parsed.wingThrustMin)
            {
                return fail(named("flappingBounds[THRUST_MIN]",
                                  "must be >= 2 * flappingAllocation[WING_MIN]"));
            }
            if (parsed.thrustMax > 2.0 * parsed.wingThrustMax)
            {
                return fail(named("flappingBounds[THRUST_MAX]",
                                  "must be <= 2 * flappingAllocation[WING_MAX]"));
            }

            parsed.weightWingThrust = weightWingThrust;

            cfg = parsed;
            reason.clear();
            return true;
        }
    };

    // Constraint report for one sampled trajectory. Every `viola*` entry is the
    // raw exceedance in its own unit and is 0 when the bound holds.
    // `maxViola` is the largest exceedance relative to its own bound, so
    // quantities with different units can still be ranked; it is 0 exactly when
    // the trajectory satisfies every enabled constraint.
    struct FlappingPenaltyReport
    {
        // Optional research functional; zero when the existing path is used.
        double additionalCost = 0.0;
        double additionalMaxViola = 0.0;
        double cost = 0.0;      // cost the caller was charged: the penalty alone
                                // from evaluatePenalty(), the whole objective from
                                // evaluateObjective()
        int nodeCount = 0;      // integration nodes examined
        int degenerateNodes = 0;// nodes where the map reported no attitude

        double maxViola = 0.0;  // largest relative exceedance
        FlappingViolation worst = FlappingViolation::None;
        int worstPiece = -1;
        double worstLocalTime = 0.0;

        double violaCorridor = 0.0;   // [m] outward half-space violation
        double violaSpeed = 0.0;      // [m/s]  ||v|| - v_max
        double violaThrustLow = 0.0;  // [N]    T_min - T
        double violaThrustHigh = 0.0; // [N]    T - T_max
        double violaOmega = 0.0;      // [rad/s]|omega_i| - omega_max_i, worst axis
        double violaGamma = 0.0;      // [N m]  |Gamma_i| - Gamma_max_i, worst axis
        double violaTilt = 0.0;       // [rad]  tilt - tilt_max (not saturated)

        double peakSpeed = 0.0;       // [m/s]
        double peakThrust = 0.0;      // [N]
        double minThrust = 0.0;       // [N]
        double peakOmega = 0.0;       // [rad/s]
        double peakGamma = 0.0;       // [N m]
        double peakTilt = 0.0;        // [rad]

        // Where the cost went: each term that contributes to `cost` above,
        // accumulated with the same trapezoidal weight (`node * step`). The
        // eight entries sum to the constraint part of `cost` up to
        // floating-point rounding: all of `cost` when the report came from
        // evaluatePenalty() or reportViolations(), and the penalty term alone
        // when it came from evaluateObjective(), whose `cost` also carries the
        // MINCO energy and the time cost. A term whose weight is zero stays
        // exactly 0.
        double penaltyCorridor = 0.0;
        double penaltyVelocity = 0.0;
        double penaltyThrustLow = 0.0;
        double penaltyThrustHigh = 0.0;
        double penaltyOmega = 0.0;
        double penaltyGamma = 0.0;
        double penaltyTilt = 0.0;
        double penaltyDegenerate = 0.0;

        // --- left/right wing thrust allocation -------------------------------
        // One forward() call per node gives T = thrust and Gamma_x = gamma(0);
        // these are the split T_L = 0.5 (T - Gamma_x / l_eff),
        // T_R = 0.5 (T + Gamma_x / l_eff) that the per-side bounds apply to.
        //
        // Observation ranges, over the nodes where the split is defined (a
        // node without a usable attitude has no Gamma_x and is not measured):
        double minWingLeft = 0.0;   // [N]
        double maxWingLeft = 0.0;   // [N]
        double minWingRight = 0.0;  // [N]
        double maxWingRight = 0.0;  // [N]
        int wingNodes = 0;          // nodes the split was evaluated at

        // Raw per-side bound exceedances [N], each 0 when its bound holds. The
        // four are independent: a node can violate only one side, or both.
        double violaWingLeftLow = 0.0;   // F_min - T_L
        double violaWingLeftHigh = 0.0;  // T_L - F_max
        double violaWingRightLow = 0.0;  // F_min - T_R
        double violaWingRightHigh = 0.0; // T_R - F_max

        // The worst of the four, in its own [N] unit, and which one it was.
        // These two mirror violaGamma/worst for the single-constraint fields:
        // the aggregate above ranks every enabled constraint by relative
        // exceedance, this one is the allocation set's own reading.
        double violaWing = 0.0;               // [N]
        FlappingViolation worstWing = FlappingViolation::None;
        double wingWorstRelative = 0.0;       // dimensionless, wingMaxViola's value

        // The allocation's share of `cost`, one entry per per-side bound, with
        // the same trapezoidal weight as the penalty_* block above. They sum to
        // the allocation term inside `cost`.
        double penaltyWingLeftLow = 0.0;
        double penaltyWingLeftHigh = 0.0;
        double penaltyWingRightLow = 0.0;
        double penaltyWingRightHigh = 0.0;

        // The configuration the report was measured against, carried so a
        // reader can tell T_L/T_R from the raw total thrust without the config
        // file at hand. source: this report's own config, never the plant.
        double rollArm = 0.0;         // [m]
        double wingThrustMin = 0.0;   // [N]
        double wingThrustMax = 0.0;   // [N]
        double weightWingThrust = 0.0;

        // Relative exceedance of the allocation set: the largest of the four
        // per-side exceedances over its own bound, using F_min for the lower
        // bounds and F_max for the upper ones. When F_min is exactly 0 the
        // lower bounds are charged against F_max instead, so a negative wing
        // thrust (a physically impossible command) is still reported rather
        // than divided by zero. `wingWorstRelative`, filled by the penalty, is
        // this quantity; the method recomputes it from the stored raw
        // exceedances and the configuration the report carries.
        inline double wingMaxViola() const
        {
            const double lowScale = wingThrustMin > 0.0
                                        ? 1.0 / wingThrustMin
                                        : (wingThrustMax > 0.0
                                               ? 1.0 / wingThrustMax
                                               : 0.0);
            const double highScale =
                wingThrustMax > 0.0 ? 1.0 / wingThrustMax : 0.0;
            double worst = 0.0;
            for (const double entry : {violaWingLeftLow * lowScale,
                                       violaWingLeftHigh * highScale,
                                       violaWingRightLow * lowScale,
                                       violaWingRightHigh * highScale})
            {
                if (entry > worst)
                {
                    worst = entry;
                }
            }
            return worst;
        }

        // The joint allocation set is satisfied only when the split actually
        // covered the whole trajectory and no per-side bound was exceeded:
        //
        //   nodeCount > 0            the report examined something
        //   degenerateNodes == 0     no node was refused an attitude
        //   wingNodes == nodeCount   every node's split was measured
        //   wingMaxViola() <= tol    no per-side bound was exceeded
        //
        // `wingMaxViola() <= tolerance` alone is not the verdict: a node the
        // map refused an attitude for has no Gamma_x, so it is never split,
        // which leaves wingNodes short of nodeCount and maxViola at 0. Reading
        // only the relative exceedance would then call a fully degenerate
        // trajectory (degenerateNodes == nodeCount, wingNodes == 0) feasible.
        // There is no node-count exception: a node with an attitude always
        // yields a (T, Gamma_x) split, so wingNodes == nodeCount holds for every
        // fully-measured trajectory.
        inline bool allocationFeasible(const double tolerance = 0.0) const
        {
            return nodeCount > 0 && degenerateNodes == 0 &&
                   wingNodes == nodeCount && wingMaxViola() <= tolerance;
        }

        inline bool feasible(const double tolerance = 0.0) const
        {
            return maxViola <= tolerance;
        }

        inline std::string describe() const
        {
            std::ostringstream stream;
            stream << "nodes=" << nodeCount
                   << " degenerate=" << degenerateNodes
                   << " cost=" << cost
                   << " maxViola=" << maxViola
                   << " worst=" << flappingViolationName(worst)
                   << " @" << worstPiece << ":" << worstLocalTime
                   << " | speed<=" << peakSpeed
                   << " thrust=[" << minThrust << "," << peakThrust << "]"
                   << " omega<=" << peakOmega
                   << " gamma<=" << peakGamma
                   << " tilt<=" << peakTilt
                   << " | viola v=" << violaSpeed
                   << " Tlo=" << violaThrustLow
                   << " Thi=" << violaThrustHigh
                   << " omg=" << violaOmega
                   << " gam=" << violaGamma
                   << " tilt=" << violaTilt
                   << " pos=" << violaCorridor
                   << " | wing L=[" << minWingLeft << "," << maxWingLeft
                   << "] R=[" << minWingRight << "," << maxWingRight
                   << "] viola Llo=" << violaWingLeftLow
                   << " Lhi=" << violaWingLeftHigh
                   << " Rlo=" << violaWingRightLow
                   << " Rhi=" << violaWingRightHigh
                   << " worst=" << flappingViolationName(worstWing);
            return stream.str();
        }
    };

    // -----------------------------------------------------------------------
    // Simplified flapping-wing planner.
    // -----------------------------------------------------------------------
    class GCOPTER_PolytopeSFC_Flapping
    {
    public:
        using AdditionalPenalty = std::function<double(
            const Eigen::VectorXd &, const Eigen::MatrixX3d &,
            Eigen::VectorXd &, Eigen::MatrixX3d &, double &)>;
        typedef Eigen::Matrix3Xd PolyhedronV;
        typedef Eigen::MatrixX4d PolyhedronH;
        typedef std::vector<PolyhedronV> PolyhedraV;
        typedef std::vector<PolyhedronH> PolyhedraH;

    private:
        // The planner optimises a 3-D position flat output, so yaw is not a
        // decision variable and is held at zero, exactly as the quadrotor
        // penalty does. omega_z is then the yaw rate the attitude chain implies,
        // and the tilt limit is the tilt of the force axis away from vertical.
        static constexpr double kPlanningYaw = 0.0;

        minco::MINCO_S3NU minco;
        flapping::FlappingFlatnessMap flatmap;

        double rho;
        Eigen::Matrix3d headPVA;
        Eigen::Matrix3d tailPVA;

        PolyhedraV vPolytopes;
        PolyhedraH hPolytopes;
        Eigen::Matrix3Xd shortPath;

        Eigen::VectorXi pieceIdx;
        Eigen::VectorXi vPolyIdx;
        Eigen::VectorXi hPolyIdx;

        int polyN;
        int pieceN;

        int spatialDim;
        int temporalDim;

        double smoothEps;
        int integralRes;
        FlappingConfig flappingCfg;
        AdditionalPenalty additionalPenalty;
        Eigen::VectorXd lastOptimizedDecision;
        bool configured;
        double allocSpeed;

        lbfgs::lbfgs_parameter_t lbfgs_params;

        Eigen::Matrix3Xd points;
        Eigen::VectorXd times;
        Eigen::Matrix3Xd gradByPoints;
        Eigen::VectorXd gradByTimes;
        Eigen::MatrixX3d partialGradByCoeffs;
        Eigen::VectorXd partialGradByTimes;

    private:
        static inline void forwardT(const Eigen::VectorXd &tau,
                                    Eigen::VectorXd &T)
        {
            const int sizeTau = tau.size();
            T.resize(sizeTau);
            for (int i = 0; i < sizeTau; i++)
            {
                T(i) = tau(i) > 0.0
                           ? ((0.5 * tau(i) + 1.0) * tau(i) + 1.0)
                           : 1.0 / ((0.5 * tau(i) - 1.0) * tau(i) + 1.0);
            }
            return;
        }

        template <typename EIGENVEC>
        static inline void backwardT(const Eigen::VectorXd &T,
                                     EIGENVEC &tau)
        {
            const int sizeT = T.size();
            tau.resize(sizeT);
            for (int i = 0; i < sizeT; i++)
            {
                tau(i) = T(i) > 1.0
                             ? (sqrt(2.0 * T(i) - 1.0) - 1.0)
                             : (1.0 - sqrt(2.0 / T(i) - 1.0));
            }

            return;
        }

        template <typename EIGENVEC>
        static inline void backwardGradT(const Eigen::VectorXd &tau,
                                         const Eigen::VectorXd &gradT,
                                         EIGENVEC &gradTau)
        {
            const int sizeTau = tau.size();
            gradTau.resize(sizeTau);
            double denSqrt;
            for (int i = 0; i < sizeTau; i++)
            {
                if (tau(i) > 0)
                {
                    gradTau(i) = gradT(i) * (tau(i) + 1.0);
                }
                else
                {
                    denSqrt = (0.5 * tau(i) - 1.0) * tau(i) + 1.0;
                    gradTau(i) = gradT(i) * (1.0 - tau(i)) / (denSqrt * denSqrt);
                }
            }

            return;
        }

        static inline void forwardP(const Eigen::VectorXd &xi,
                                    const Eigen::VectorXi &vIdx,
                                    const PolyhedraV &vPolys,
                                    Eigen::Matrix3Xd &P)
        {
            const int sizeP = vIdx.size();
            P.resize(3, sizeP);
            Eigen::VectorXd q;
            for (int i = 0, j = 0, k, l; i < sizeP; i++, j += k)
            {
                l = vIdx(i);
                k = vPolys[l].cols();
                q = xi.segment(j, k).normalized().head(k - 1);
                P.col(i) = vPolys[l].rightCols(k - 1) * q.cwiseProduct(q) +
                           vPolys[l].col(0);
            }
            return;
        }

        static inline double costTinyNLS(void *ptr,
                                         const Eigen::VectorXd &xi,
                                         Eigen::VectorXd &gradXi)
        {
            const int n = xi.size();
            const Eigen::Matrix3Xd &ovPoly = *(Eigen::Matrix3Xd *)ptr;

            const double sqrNormXi = xi.squaredNorm();
            const double invNormXi = 1.0 / sqrt(sqrNormXi);
            const Eigen::VectorXd unitXi = xi * invNormXi;
            const Eigen::VectorXd r = unitXi.head(n - 1);
            const Eigen::Vector3d delta = ovPoly.rightCols(n - 1) * r.cwiseProduct(r) +
                                          ovPoly.col(1) - ovPoly.col(0);

            double cost = delta.squaredNorm();
            gradXi.head(n - 1) = (ovPoly.rightCols(n - 1).transpose() * (2 * delta)).array() *
                                 r.array() * 2.0;
            gradXi(n - 1) = 0.0;
            gradXi = (gradXi - unitXi.dot(gradXi) * unitXi).eval() * invNormXi;

            const double sqrNormViolation = sqrNormXi - 1.0;
            if (sqrNormViolation > 0.0)
            {
                double c = sqrNormViolation * sqrNormViolation;
                const double dc = 3.0 * c;
                c *= sqrNormViolation;
                cost += c;
                gradXi += dc * 2.0 * xi;
            }

            return cost;
        }

        template <typename EIGENVEC>
        static inline void backwardP(const Eigen::Matrix3Xd &P,
                                     const Eigen::VectorXi &vIdx,
                                     const PolyhedraV &vPolys,
                                     EIGENVEC &xi)
        {
            const int sizeP = P.cols();

            double minSqrD;
            lbfgs::lbfgs_parameter_t tiny_nls_params;
            tiny_nls_params.past = 0;
            tiny_nls_params.delta = 1.0e-5;
            tiny_nls_params.g_epsilon = FLT_EPSILON;
            tiny_nls_params.max_iterations = 128;

            Eigen::Matrix3Xd ovPoly;
            for (int i = 0, j = 0, k, l; i < sizeP; i++, j += k)
            {
                l = vIdx(i);
                k = vPolys[l].cols();

                ovPoly.resize(3, k + 1);
                ovPoly.col(0) = P.col(i);
                ovPoly.rightCols(k) = vPolys[l];
                Eigen::VectorXd x(k);
                x.setConstant(sqrt(1.0 / k));
                lbfgs::lbfgs_optimize(x,
                                      minSqrD,
                                      &GCOPTER_PolytopeSFC_Flapping::costTinyNLS,
                                      nullptr,
                                      nullptr,
                                      &ovPoly,
                                      tiny_nls_params);

                xi.segment(j, k) = x;
            }

            return;
        }

        template <typename EIGENVEC>
        static inline void backwardGradP(const Eigen::VectorXd &xi,
                                         const Eigen::VectorXi &vIdx,
                                         const PolyhedraV &vPolys,
                                         const Eigen::Matrix3Xd &gradP,
                                         EIGENVEC &gradXi)
        {
            const int sizeP = vIdx.size();
            gradXi.resize(xi.size());

            double normInv;
            Eigen::VectorXd q, gradQ, unitQ;
            for (int i = 0, j = 0, k, l; i < sizeP; i++, j += k)
            {
                l = vIdx(i);
                k = vPolys[l].cols();
                q = xi.segment(j, k);
                normInv = 1.0 / q.norm();
                unitQ = q * normInv;
                gradQ.resize(k);
                gradQ.head(k - 1) = (vPolys[l].rightCols(k - 1).transpose() * gradP.col(i)).array() *
                                    unitQ.head(k - 1).array() * 2.0;
                gradQ(k - 1) = 0.0;
                gradXi.segment(j, k) = (gradQ - unitQ * unitQ.dot(gradQ)) * normInv;
            }

            return;
        }

        template <typename EIGENVEC>
        static inline void normRetrictionLayer(const Eigen::VectorXd &xi,
                                               const Eigen::VectorXi &vIdx,
                                               const PolyhedraV &vPolys,
                                               double &cost,
                                               EIGENVEC &gradXi)
        {
            const int sizeP = vIdx.size();
            gradXi.resize(xi.size());

            double sqrNormQ, sqrNormViolation, c, dc;
            Eigen::VectorXd q;
            for (int i = 0, j = 0, k; i < sizeP; i++, j += k)
            {
                k = vPolys[vIdx(i)].cols();

                q = xi.segment(j, k);
                sqrNormQ = q.squaredNorm();
                sqrNormViolation = sqrNormQ - 1.0;
                if (sqrNormViolation > 0.0)
                {
                    c = sqrNormViolation * sqrNormViolation;
                    dc = 3.0 * c;
                    c *= sqrNormViolation;
                    cost += c;
                    gradXi.segment(j, k) += dc * 2.0 * q;
                }
            }

            return;
        }

        static inline bool smoothedL1(const double &x,
                                      const double &mu,
                                      double &f,
                                      double &df)
        {
            if (x < 0.0)
            {
                return false;
            }
            else if (x > mu)
            {
                f = x - 0.5 * mu;
                df = 1.0;
                return true;
            }
            else
            {
                const double xdmu = x / mu;
                const double sqrxdmu = xdmu * xdmu;
                const double mumxd2 = mu - 0.5 * x;
                f = mumxd2 * sqrxdmu * xdmu;
                df = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
                return true;
            }
        }

        // -------------------------------------------------------------------
        // The flapping constraint penalty.
        //
        // Sampled at the same trapezoidal nodes the quadrotor penalty uses, so
        // the quadrature and the time gradient keep the same structure. Two
        // things differ, both forced by the model:
        //
        //  * the map consumes snap, so the coefficient and time gradients carry
        //    a beta4 term and the integrand's own time dependence reaches
        //    crackle = d(snap)/ds = c^T beta5 with beta5 = [0,0,0,0,0,120];
        //  * the constraint set is the flapping one (speed, thrust interval,
        //    per-axis omega and Gamma, optional tilt) rather than the
        //    quadrotor's [v, omega, tilt, thrust] packing.
        //
        // Trapezoidal weights: with h = T_i / N and nodes s_j = j h,
        //     integral ~= h * [0.5 f_0 + f_1 + ... + f_{N-1} + 0.5 f_N],
        // which is `node * step` with node = 0.5 at j = 0 and j = N and 1
        // otherwise. Differentiating w.r.t. T_i gives two contributions: the
        // change of the weights, h' = 1/N, times the integrand; and the change
        // of the node positions, ds_j/dT_i = j/N = alpha, times the integrand's
        // own s-derivative. Both are below, and the s-derivative now runs to
        // crackle because snap is an input of the map.
        // -------------------------------------------------------------------
        static inline void attachPenaltyFunctional(const Eigen::VectorXd &T,
                                                   const Eigen::MatrixX3d &coeffs,
                                                   const Eigen::VectorXi &hIdx,
                                                   const PolyhedraH &hPolys,
                                                   const double &smoothFactor,
                                                   const int &integralResolution,
                                                   const FlappingConfig &cfg,
                                                   const flapping::FlappingFlatnessMap &flatMap,
                                                   double &cost,
                                                   Eigen::VectorXd &gradT,
                                                   Eigen::MatrixX3d &gradC,
                                                   FlappingPenaltyReport *report)
        {
            const double velSqrMax = cfg.velMax * cfg.velMax;
            const double cosTiltMax = std::cos(cfg.tiltMax);
            const double tiltScale = 1.0 / cfg.tiltMax;
            const double velScale = 1.0 / cfg.velMax;
            const double thrustLowScale =
                cfg.thrustMin > 0.0 ? 1.0 / cfg.thrustMin : 0.0;
            const double thrustHighScale = 1.0 / cfg.thrustMax;
            const double omegaScale =
                1.0 / cfg.omegaMax.maxCoeff();
            const double gammaScale =
                1.0 / cfg.gammaMax.maxCoeff();
            // The per-side allocation bounds are normalised the same way the
            // total-thrust bounds are: F_min for the lower bounds, F_max for the
            // upper ones. When F_min is exactly 0 the lower bounds fall back to
            // F_max, so a negative wing thrust (a physically impossible command)
            // is still a finite relative exceedance rather than a divide by zero.
            const double wingLowScale =
                cfg.wingThrustMin > 0.0
                    ? 1.0 / cfg.wingThrustMin
                    : (cfg.wingThrustMax > 0.0 ? 1.0 / cfg.wingThrustMax : 0.0);
            const double wingHighScale =
                cfg.wingThrustMax > 0.0 ? 1.0 / cfg.wingThrustMax : 0.0;

            if (report != nullptr)
            {
                // The bounds the allocation readings are measured against,
                // carried with the report so a reader can split T_L/T_R without
                // the config file. This is this run's own validated config.
                report->rollArm = cfg.rollArm;
                report->wingThrustMin = cfg.wingThrustMin;
                report->wingThrustMax = cfg.wingThrustMax;
                report->weightWingThrust = cfg.weightWingThrust;
            }

            const int pieceNum = T.size();
            const double integralFrac = 1.0 / integralResolution;

            Eigen::Vector3d pos, vel, acc, jer, sna, cra;
            Eigen::Vector3d totalGradPos, totalGradVel, totalGradAcc;
            Eigen::Vector3d totalGradJer, totalGradSnap;
            Eigen::Vector3d gradVelTerm, gradAccTilt;
            Eigen::Vector3d outerNormal;
            double cosTheta;
            Eigen::Vector3d omg, gamma;
            double gradThr;
            Eigen::Vector4d gradQuat;
            Eigen::Vector3d gradOmg, gradGamma;
            double step, alpha, node, pena;
            double violaPos, violaVelSqr, violaTLo, violaTHi, violaOmg, violaGam;
            double violaTiltCos, violaTiltAngle;
            double violaWingLeftLow, violaWingLeftHigh;
            double violaWingRightLow, violaWingRightHigh;
            double tLeft, tRight, gradThrWing, gradGammaWing;
            bool wingUsable;
            double penaPos, penaVel, penaTLo, penaTHi, penaOmg, penaGam, penaTilt;
            // Per-term split of `pena`, so the report can say which constraint
            // the cost came from.
            double penaTermCorridor, penaTermVelocity, penaTermThrustLow;
            double penaTermThrustHigh, penaTermOmega, penaTermGamma;
            double penaTermTilt, penaTermDegenerate;
            // The four per-side wing-thrust split terms.
            double penaTermWingLeftLow, penaTermWingLeftHigh;
            double penaTermWingRightLow, penaTermWingRightHigh;
            double dPos, dVel, dTLo, dTHi, dOmg, dGam, dTilt;
            double dWingLeftLow, dWingLeftHigh, dWingRightLow, dWingRightHigh;
            double s1, s2, s3, s4, s5;
            Eigen::Matrix<double, 6, 1> beta0, beta1, beta2, beta3, beta4, beta5;
            int K, L;

            for (int i = 0; i < pieceNum; i++)
            {
                const Eigen::Matrix<double, 6, 3> &c = coeffs.block<6, 3>(i * 6, 0);

                // beta5 = d^5 (beta0) / ds^5; crackle is constant inside a
                // fifth-order piece, so it is hoisted out of the node loop.
                beta5 << 0.0, 0.0, 0.0, 0.0, 0.0, 120.0;
                cra = c.transpose() * beta5;

                step = T(i) * integralFrac;
                for (int j = 0; j <= integralResolution; j++)
                {
                    s1 = j * step;
                    s2 = s1 * s1;
                    s3 = s2 * s1;
                    s4 = s2 * s2;
                    s5 = s4 * s1;
                    beta0(0) = 1.0, beta0(1) = s1, beta0(2) = s2, beta0(3) = s3, beta0(4) = s4, beta0(5) = s5;
                    beta1(0) = 0.0, beta1(1) = 1.0, beta1(2) = 2.0 * s1, beta1(3) = 3.0 * s2, beta1(4) = 4.0 * s3, beta1(5) = 5.0 * s4;
                    beta2(0) = 0.0, beta2(1) = 0.0, beta2(2) = 2.0, beta2(3) = 6.0 * s1, beta2(4) = 12.0 * s2, beta2(5) = 20.0 * s3;
                    beta3(0) = 0.0, beta3(1) = 0.0, beta3(2) = 0.0, beta3(3) = 6.0, beta3(4) = 24.0 * s1, beta3(5) = 60.0 * s2;
                    beta4(0) = 0.0, beta4(1) = 0.0, beta4(2) = 0.0, beta4(3) = 0.0, beta4(4) = 24.0, beta4(5) = 120.0 * s1;
                    pos = c.transpose() * beta0;
                    vel = c.transpose() * beta1;
                    acc = c.transpose() * beta2;
                    jer = c.transpose() * beta3;
                    sna = c.transpose() * beta4;

                    const flapping::ForwardResult fr = flatMap.forward(
                        vel, acc, jer, sna, kPlanningYaw, 0.0, 0.0);

                    // The map always reports T = m ||a - g_w||, even when it
                    // refuses to build an attitude; the attitude-dependent
                    // outputs are NaN exactly when no attitude exists, so both
                    // predicates below are false in both degenerate statuses.
                    //
                    // They ask different questions, and only fr.ok() is the
                    // authority on whether an attitude exists:
                    //   * dirUsable     -- is z = u / ||u|| defined? True for
                    //                      AttitudeSingular (T > thrust_eps) and
                    //                      for Ok; the tilt penalty and the
                    //                      thrust fallback below need only z.
                    //   * rateUsable    -- are omega and Gamma finite? Same
                    //                      test as fr.ok() for this map, spelled
                    //                      as a value check so a NaN can never
                    //                      reach `cost` through the axis terms.
                    const bool dirUsable = flapping::detail::isFinite(fr.force_dir);
                    const bool rateUsable =
                        flapping::detail::isFinite(fr.omega) &&
                        flapping::detail::isFinite(fr.gamma);

                    gradThr = 0.0;
                    gradQuat.setZero();
                    gradOmg.setZero();
                    gradGamma.setZero();
                    gradVelTerm.setZero();
                    gradAccTilt.setZero();
                    pena = 0.0;
                    penaTermCorridor = 0.0;
                    penaTermVelocity = 0.0;
                    penaTermThrustLow = 0.0;
                    penaTermThrustHigh = 0.0;
                    penaTermOmega = 0.0;
                    penaTermGamma = 0.0;
                    penaTermTilt = 0.0;
                    penaTermDegenerate = 0.0;
                    penaTermWingLeftLow = 0.0;
                    penaTermWingLeftHigh = 0.0;
                    penaTermWingRightLow = 0.0;
                    penaTermWingRightHigh = 0.0;
                    violaTLo = 0.0;
                    violaTHi = 0.0;
                    violaOmg = 0.0;
                    violaGam = 0.0;
                    violaWingLeftLow = 0.0;
                    violaWingLeftHigh = 0.0;
                    violaWingRightLow = 0.0;
                    violaWingRightHigh = 0.0;
                    wingUsable = false;
                    violaTiltAngle = 0.0;

                    // --- position corridor (unchanged from the quadrotor map) --
                    totalGradPos.setZero();
                    L = hIdx(i);
                    K = hPolys[L].rows();
                    violaPos = 0.0;
                    for (int k = 0; k < K; k++)
                    {
                        outerNormal = hPolys[L].block<1, 3>(k, 0);
                        const double violaK = outerNormal.dot(pos) + hPolys[L](k, 3);
                        if (violaK > violaPos)
                        {
                            violaPos = violaK;
                        }
                        if (smoothedL1(violaK, smoothFactor, penaPos, dPos))
                        {
                            totalGradPos += cfg.weightPosition * dPos * outerNormal;
                            pena += cfg.weightPosition * penaPos;
                            penaTermCorridor += cfg.weightPosition * penaPos;
                        }
                    }

                    // --- ||v|| <= v_max ---------------------------------------
                    violaVelSqr = vel.squaredNorm() - velSqrMax;
                    if (smoothedL1(violaVelSqr, smoothFactor, penaVel, dVel))
                    {
                        gradVelTerm = cfg.weightVelocity * dVel * 2.0 * vel;
                        pena += cfg.weightVelocity * penaVel;
                        penaTermVelocity += cfg.weightVelocity * penaVel;
                    }

                    // --- T_min <= T <= T_max ----------------------------------
                    violaTLo = cfg.thrustMin - fr.thrust;
                    if (smoothedL1(violaTLo, smoothFactor, penaTLo, dTLo))
                    {
                        gradThr += -cfg.weightThrust * dTLo;
                        pena += cfg.weightThrust * penaTLo;
                        penaTermThrustLow += cfg.weightThrust * penaTLo;
                    }
                    violaTHi = fr.thrust - cfg.thrustMax;
                    if (smoothedL1(violaTHi, smoothFactor, penaTHi, dTHi))
                    {
                        gradThr += cfg.weightThrust * dTHi;
                        pena += cfg.weightThrust * penaTHi;
                        penaTermThrustHigh += cfg.weightThrust * penaTHi;
                    }

                    // --- F_min <= T_L, T_R <= F_max --------------------------
                    // The left/right split of the SAME forward() call's total
                    // thrust and body roll moment:
                    //     T_L = 0.5 (T - Gamma_x / l_eff)
                    //     T_R = 0.5 (T + Gamma_x / l_eff)
                    // and the four per-side violations. Gamma_x is gamma(0), so
                    // this is a fourth-order constraint through the snap.
                    //
                    // Only a node with the whole attitude (fr.ok()) is split: a
                    // node with no Gamma_x has no T_L/T_R to compare, gets no
                    // fabricated values, and keeps the existing degenerate
                    // charge. In that case dirUsable here would still be NaN
                    // for gamma, so it could not be split anyway.
                    if (fr.ok())
                    {
                        const double invArm = 1.0 / cfg.rollArm;
                        const double halfArm = 0.5 * invArm;
                        tLeft = 0.5 * (fr.thrust - fr.gamma(0) * invArm);
                        tRight = 0.5 * (fr.thrust + fr.gamma(0) * invArm);
                        wingUsable = true;

                        violaWingLeftLow = cfg.wingThrustMin - tLeft;
                        if (smoothedL1(violaWingLeftLow, smoothFactor,
                                       penaTLo, dWingLeftLow))
                        {
                            gradThrWing = -0.5 * cfg.weightWingThrust * dWingLeftLow;
                            gradGammaWing = halfArm * cfg.weightWingThrust * dWingLeftLow;
                            pena += cfg.weightWingThrust * penaTLo;
                            penaTermWingLeftLow += cfg.weightWingThrust * penaTLo;
                        }
                        else
                        {
                            gradThrWing = 0.0;
                            gradGammaWing = 0.0;
                        }
                        violaWingLeftHigh = tLeft - cfg.wingThrustMax;
                        if (smoothedL1(violaWingLeftHigh, smoothFactor,
                                       penaTHi, dWingLeftHigh))
                        {
                            gradThrWing += 0.5 * cfg.weightWingThrust * dWingLeftHigh;
                            gradGammaWing -= halfArm * cfg.weightWingThrust * dWingLeftHigh;
                            pena += cfg.weightWingThrust * penaTHi;
                            penaTermWingLeftHigh += cfg.weightWingThrust * penaTHi;
                        }
                        gradThr += gradThrWing;
                        gradGamma(0) += gradGammaWing;

                        violaWingRightLow = cfg.wingThrustMin - tRight;
                        if (smoothedL1(violaWingRightLow, smoothFactor,
                                       penaTLo, dWingRightLow))
                        {
                            gradThrWing = -0.5 * cfg.weightWingThrust * dWingRightLow;
                            gradGammaWing = -halfArm * cfg.weightWingThrust * dWingRightLow;
                            pena += cfg.weightWingThrust * penaTLo;
                            penaTermWingRightLow += cfg.weightWingThrust * penaTLo;
                        }
                        else
                        {
                            gradThrWing = 0.0;
                            gradGammaWing = 0.0;
                        }
                        violaWingRightHigh = tRight - cfg.wingThrustMax;
                        if (smoothedL1(violaWingRightHigh, smoothFactor,
                                       penaTHi, dWingRightHigh))
                        {
                            gradThrWing += 0.5 * cfg.weightWingThrust * dWingRightHigh;
                            gradGammaWing += halfArm * cfg.weightWingThrust * dWingRightHigh;
                            pena += cfg.weightWingThrust * penaTHi;
                            penaTermWingRightHigh += cfg.weightWingThrust * penaTHi;
                        }
                        gradThr += gradThrWing;
                        gradGamma(0) += gradGammaWing;
                    }

                    // --- per-axis |omega_i| <= omega_max_i, |Gamma_i| <= ... ---
                    if (rateUsable)
                    {
                        omg = fr.omega;
                        gamma = fr.gamma;
                        for (int a = 0; a < 3; a++)
                        {
                            const double violaAxis = std::fabs(omg(a)) - cfg.omegaMax(a);
                            if (violaAxis > violaOmg)
                            {
                                violaOmg = violaAxis;
                            }
                            if (smoothedL1(violaAxis, smoothFactor, penaOmg, dOmg))
                            {
                                gradOmg(a) += cfg.weightOmega * dOmg *
                                              (omg(a) >= 0.0 ? 1.0 : -1.0);
                                pena += cfg.weightOmega * penaOmg;
                                penaTermOmega += cfg.weightOmega * penaOmg;
                            }

                            const double violaMoment = std::fabs(gamma(a)) - cfg.gammaMax(a);
                            if (violaMoment > violaGam)
                            {
                                violaGam = violaMoment;
                            }
                            if (smoothedL1(violaMoment, smoothFactor, penaGam, dGam))
                            {
                                gradGamma(a) += cfg.weightGamma * dGam *
                                                (gamma(a) >= 0.0 ? 1.0 : -1.0);
                                pena += cfg.weightGamma * penaGam;
                                penaTermGamma += cfg.weightGamma * penaGam;
                            }
                        }
                    }
                    else
                    {
                        // No attitude means no omega and no Gamma to compare
                        // against their limits. Charge a constant instead of
                        // letting the node pass for free. Both degenerate
                        // statuses (ThrustBelowEpsilon, AttitudeSingular) land
                        // here, so the cost and the report keep showing them
                        // even though no analytic gradient exists.
                        pena += cfg.weightDegenerate;
                        penaTermDegenerate += cfg.weightDegenerate;
                        if (report != nullptr)
                        {
                            report->degenerateNodes++;
                        }
                    }

                    // --- optional tilt limit ----------------------------------
                    // Measured on cos(tilt) = z_z rather than on the angle, so
                    // the gradient stays bounded when the vehicle points near
                    // straight down. The report converts back to radians.
                    if (cfg.tiltEnabled && dirUsable)
                    {
                        cosTheta = fr.force_dir(2);
                        violaTiltCos = cosTiltMax - cosTheta;
                        if (cosTheta < cosTiltMax)
                        {
                            violaTiltAngle =
                                std::acos(cosTheta < -1.0 ? -1.0 : cosTheta) -
                                cfg.tiltMax;
                        }
                        if (smoothedL1(violaTiltCos, smoothFactor, penaTilt, dTilt))
                        {
                            // cosTheta = u_z / ||u|| with u = a - g_w, so
                            // d cosTheta / d a = (e_z - cosTheta z) / ||u|| and
                            // ||u|| = T / m. Taken analytically instead of
                            // through the quaternion, because the quaternion
                            // path is undefined exactly where the map reports
                            // AttitudeSingular and the tilt penalty matters
                            // most.
                            gradAccTilt = -cfg.weightTilt * dTilt *
                                          (Eigen::Vector3d(0.0, 0.0, 1.0) -
                                           cosTheta * fr.force_dir) *
                                          (cfg.mass / fr.thrust);
                            pena += cfg.weightTilt * penaTilt;
                            penaTermTilt += cfg.weightTilt * penaTilt;
                        }
                    }
                    else if (cfg.tiltEnabled)
                    {
                        // Attitude undefined: report the tilt as violated
                        // rather than as satisfied.
                        violaTiltAngle = 0.5 * M_PI - cfg.tiltMax;
                        pena += cfg.weightTilt * (cosTiltMax + 1.0);
                        penaTermTilt += cfg.weightTilt * (cosTiltMax + 1.0);
                    }

                    // --- chain the non-map terms and run the reverse pass ----
                    totalGradVel = gradVelTerm;
                    totalGradAcc = gradAccTilt;
                    totalGradJer.setZero();
                    totalGradSnap.setZero();
                    // fr.ok() -- not dirUsable -- decides whether there is a
                    // reverse pass. AttitudeSingular still reports a finite
                    // force direction (T > thrust_eps), but backward() returns
                    // NaN for every input whenever forward() was not Ok, and
                    // those NaNs would land in gradC and gradT below.
                    if (fr.ok())
                    {
                        flapping::BackwardResult back = flatMap.backward(
                            vel, acc, jer, sna, kPlanningYaw, 0.0, 0.0, gradThr,
                            gradQuat, gradOmg, Eigen::Vector3d::Zero(), gradGamma);
                        totalGradVel += back.vel;
                        totalGradAcc += back.acc;
                        totalGradJer = back.jer;
                        totalGradSnap = back.snap;
                    }
                    else if (dirUsable)
                    {
                        // AttitudeSingular: no attitude, so the map contributes
                        // no gradient, but the thrust is still differentiable on
                        // its own. T = m ||a - g_w|| gives
                        // dT/da = m (a - g_w) / ||a - g_w||, and dirUsable means
                        // ||a - g_w|| > thrust_eps / m > 0, so the division is
                        // safe. The thrust constraint therefore keeps pushing the
                        // node back towards a state where the attitude exists.
                        const Eigen::Vector3d u =
                            acc + Eigen::Vector3d(0.0, 0.0, cfg.gravity);
                        totalGradAcc += (gradThr * cfg.mass / u.norm()) * u;
                    }
                    // ThrustBelowEpsilon: ||a - g_w|| is not differentiable at
                    // zero and the force direction does not exist, so nothing
                    // analytic can be propagated. The node keeps the constant
                    // degenerate penalty and the count charged above.

                    node = (j == 0 || j == integralResolution) ? 0.5 : 1.0;
                    alpha = j * integralFrac;
                    gradC.block<6, 3>(i * 6, 0) += (beta0 * totalGradPos.transpose() +
                                                    beta1 * totalGradVel.transpose() +
                                                    beta2 * totalGradAcc.transpose() +
                                                    beta3 * totalGradJer.transpose() +
                                                    beta4 * totalGradSnap.transpose()) *
                                                   node * step;
                    gradT(i) += (totalGradPos.dot(vel) +
                                 totalGradVel.dot(acc) +
                                 totalGradAcc.dot(jer) +
                                 totalGradJer.dot(sna) +
                                 totalGradSnap.dot(cra)) *
                                    alpha * node * step +
                                node * integralFrac * pena;
                    cost += node * step * pena;

                    if (report != nullptr)
                    {
                        const bool firstNode = report->nodeCount == 0;
                        report->nodeCount++;
                        const double weighted = node * step;
                        report->penaltyCorridor += weighted * penaTermCorridor;
                        report->penaltyVelocity += weighted * penaTermVelocity;
                        report->penaltyThrustLow += weighted * penaTermThrustLow;
                        report->penaltyThrustHigh += weighted * penaTermThrustHigh;
                        report->penaltyOmega += weighted * penaTermOmega;
                        report->penaltyGamma += weighted * penaTermGamma;
                        report->penaltyTilt += weighted * penaTermTilt;
                        report->penaltyDegenerate += weighted * penaTermDegenerate;
                        report->penaltyWingLeftLow += weighted * penaTermWingLeftLow;
                        report->penaltyWingLeftHigh += weighted * penaTermWingLeftHigh;
                        report->penaltyWingRightLow += weighted * penaTermWingRightLow;
                        report->penaltyWingRightHigh += weighted * penaTermWingRightHigh;
                        if (violaPos > report->violaCorridor)
                        {
                            report->violaCorridor = violaPos;
                        }
                        const double speed = vel.norm();
                        if (speed > report->peakSpeed)
                        {
                            report->peakSpeed = speed;
                        }
                        if (fr.thrust > report->peakThrust)
                        {
                            report->peakThrust = fr.thrust;
                        }
                        if (firstNode || fr.thrust < report->minThrust)
                        {
                            report->minThrust = fr.thrust;
                        }
                        if (violaTLo > report->violaThrustLow)
                        {
                            report->violaThrustLow = violaTLo;
                        }
                        if (violaTHi > report->violaThrustHigh)
                        {
                            report->violaThrustHigh = violaTHi;
                        }
                        if ((speed - cfg.velMax) > report->violaSpeed)
                        {
                            report->violaSpeed = speed - cfg.velMax;
                        }
                        if (rateUsable)
                        {
                            if (omg.cwiseAbs().maxCoeff() > report->peakOmega)
                            {
                                report->peakOmega = omg.cwiseAbs().maxCoeff();
                            }
                            if (gamma.cwiseAbs().maxCoeff() > report->peakGamma)
                            {
                                report->peakGamma = gamma.cwiseAbs().maxCoeff();
                            }
                            if (violaOmg > report->violaOmega)
                            {
                                report->violaOmega = violaOmg;
                            }
                            if (violaGam > report->violaGamma)
                            {
                                report->violaGamma = violaGam;
                            }
                        }
                        if (dirUsable)
                        {
                            const double reached =
                                std::acos(fr.force_dir(2) < -1.0 ? -1.0
                                                                 : fr.force_dir(2));
                            if (reached > report->peakTilt)
                            {
                                report->peakTilt = reached;
                            }
                        }
                        if (violaTiltAngle > report->violaTilt)
                        {
                            report->violaTilt = violaTiltAngle;
                        }

                        // Left/right wing split observation. Only nodes the map
                        // built an attitude for have a T_L/T_R; a degenerate
                        // node contributes no fabricated reading.
                        if (wingUsable)
                        {
                            const bool firstWing = report->wingNodes == 0;
                            report->wingNodes++;
                            if (firstWing || tLeft < report->minWingLeft)
                            {
                                report->minWingLeft = tLeft;
                            }
                            if (firstWing || tLeft > report->maxWingLeft)
                            {
                                report->maxWingLeft = tLeft;
                            }
                            if (firstWing || tRight < report->minWingRight)
                            {
                                report->minWingRight = tRight;
                            }
                            if (firstWing || tRight > report->maxWingRight)
                            {
                                report->maxWingRight = tRight;
                            }
                            if (violaWingLeftLow > report->violaWingLeftLow)
                            {
                                report->violaWingLeftLow = violaWingLeftLow;
                            }
                            if (violaWingLeftHigh > report->violaWingLeftHigh)
                            {
                                report->violaWingLeftHigh = violaWingLeftHigh;
                            }
                            if (violaWingRightLow > report->violaWingRightLow)
                            {
                                report->violaWingRightLow = violaWingRightLow;
                            }
                            if (violaWingRightHigh > report->violaWingRightHigh)
                            {
                                report->violaWingRightHigh = violaWingRightHigh;
                            }
                            // The allocation's own worst entry: the largest
                            // relative exceedance over the four per-side bounds,
                            // reported with the same ranking the aggregate uses.
                            const double wingRelative[4] = {
                                violaWingLeftLow * wingLowScale,
                                violaWingLeftHigh * wingHighScale,
                                violaWingRightLow * wingLowScale,
                                violaWingRightHigh * wingHighScale};
                            const FlappingViolation wingKinds[4] = {
                                FlappingViolation::WingLeftLow,
                                FlappingViolation::WingLeftHigh,
                                FlappingViolation::WingRightLow,
                                FlappingViolation::WingRightHigh};
                            for (int k = 0; k < 4; k++)
                            {
                                if (wingRelative[k] > report->wingWorstRelative)
                                {
                                    report->wingWorstRelative = wingRelative[k];
                                    report->worstWing = wingKinds[k];
                                    report->violaWing =
                                        (k == 0)   ? violaWingLeftLow
                                        : (k == 1) ? violaWingLeftHigh
                                        : (k == 2) ? violaWingRightLow
                                                   : violaWingRightHigh;
                                }
                            }
                        }

                        // Rank the violations by how far past their own bound
                        // they are, so different units can be compared.
                        const double relative[12] = {
                            violaPos,
                            violaVelSqr > 0.0 ? (speed - cfg.velMax) * velScale : 0.0,
                            violaTLo * thrustLowScale,
                            violaTHi * thrustHighScale,
                            violaOmg * omegaScale,
                            violaGam * gammaScale,
                            violaTiltAngle * tiltScale,
                            rateUsable ? 0.0 : 1.0,
                            wingUsable ? violaWingLeftLow * wingLowScale : 0.0,
                            wingUsable ? violaWingLeftHigh * wingHighScale : 0.0,
                            wingUsable ? violaWingRightLow * wingLowScale : 0.0,
                            wingUsable ? violaWingRightHigh * wingHighScale : 0.0};
                        const FlappingViolation kinds[12] = {
                            FlappingViolation::Corridor,
                            FlappingViolation::Speed,
                            FlappingViolation::ThrustLow,
                            FlappingViolation::ThrustHigh,
                            FlappingViolation::Omega,
                            FlappingViolation::Gamma,
                            FlappingViolation::Tilt,
                            FlappingViolation::Degenerate,
                            FlappingViolation::WingLeftLow,
                            FlappingViolation::WingLeftHigh,
                            FlappingViolation::WingRightLow,
                            FlappingViolation::WingRightHigh};
                        for (int k = 0; k < 12; k++)
                        {
                            if (relative[k] > report->maxViola)
                            {
                                report->maxViola = relative[k];
                                report->worst = kinds[k];
                                report->worstPiece = i;
                                report->worstLocalTime = s1;
                            }
                        }
                    }
                }
            }

            if (report != nullptr)
            {
                report->cost = cost;
            }
            return;
        }

        static inline double costFunctional(void *ptr,
                                            const Eigen::VectorXd &x,
                                            Eigen::VectorXd &g)
        {
            GCOPTER_PolytopeSFC_Flapping &obj =
                *(GCOPTER_PolytopeSFC_Flapping *)ptr;
            return obj.evaluateObjective(x, g);
        }

        static inline double costDistance(void *ptr,
                                          const Eigen::VectorXd &xi,
                                          Eigen::VectorXd &gradXi)
        {
            void **dataPtrs = (void **)ptr;
            const double &dEps = *((const double *)(dataPtrs[0]));
            const Eigen::Vector3d &ini = *((const Eigen::Vector3d *)(dataPtrs[1]));
            const Eigen::Vector3d &fin = *((const Eigen::Vector3d *)(dataPtrs[2]));
            const PolyhedraV &vPolys = *((PolyhedraV *)(dataPtrs[3]));

            double cost = 0.0;
            const int overlaps = vPolys.size() / 2;

            Eigen::Matrix3Xd gradP = Eigen::Matrix3Xd::Zero(3, overlaps);
            Eigen::Vector3d a, b, d;
            Eigen::VectorXd r;
            double smoothedDistance;
            for (int i = 0, j = 0, k = 0; i <= overlaps; i++, j += k)
            {
                a = i == 0 ? ini : b;
                if (i < overlaps)
                {
                    k = vPolys[2 * i + 1].cols();
                    Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
                    r = q.normalized().head(k - 1);
                    b = vPolys[2 * i + 1].rightCols(k - 1) * r.cwiseProduct(r) +
                        vPolys[2 * i + 1].col(0);
                }
                else
                {
                    b = fin;
                }

                d = b - a;
                smoothedDistance = sqrt(d.squaredNorm() + dEps);
                cost += smoothedDistance;

                if (i < overlaps)
                {
                    gradP.col(i) += d / smoothedDistance;
                }
                if (i > 0)
                {
                    gradP.col(i - 1) -= d / smoothedDistance;
                }
            }

            Eigen::VectorXd unitQ;
            double sqrNormQ, invNormQ, sqrNormViolation, c, dc;
            for (int i = 0, j = 0, k; i < overlaps; i++, j += k)
            {
                k = vPolys[2 * i + 1].cols();
                Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
                Eigen::Map<Eigen::VectorXd> gradQ(gradXi.data() + j, k);
                sqrNormQ = q.squaredNorm();
                invNormQ = 1.0 / sqrt(sqrNormQ);
                unitQ = q * invNormQ;
                gradQ.head(k - 1) = (vPolys[2 * i + 1].rightCols(k - 1).transpose() * gradP.col(i)).array() *
                                    unitQ.head(k - 1).array() * 2.0;
                gradQ(k - 1) = 0.0;
                gradQ = (gradQ - unitQ * unitQ.dot(gradQ)).eval() * invNormQ;

                sqrNormViolation = sqrNormQ - 1.0;
                if (sqrNormViolation > 0.0)
                {
                    c = sqrNormViolation * sqrNormViolation;
                    dc = 3.0 * c;
                    c *= sqrNormViolation;
                    cost += c;
                    gradQ += dc * 2.0 * q;
                }
            }

            return cost;
        }

        static inline void getShortestPath(const Eigen::Vector3d &ini,
                                           const Eigen::Vector3d &fin,
                                           const PolyhedraV &vPolys,
                                           const double &smoothD,
                                           Eigen::Matrix3Xd &path)
        {
            const int overlaps = vPolys.size() / 2;
            Eigen::VectorXi vSizes(overlaps);
            for (int i = 0; i < overlaps; i++)
            {
                vSizes(i) = vPolys[2 * i + 1].cols();
            }
            Eigen::VectorXd xi(vSizes.sum());
            for (int i = 0, j = 0; i < overlaps; i++)
            {
                xi.segment(j, vSizes(i)).setConstant(sqrt(1.0 / vSizes(i)));
                j += vSizes(i);
            }

            double minDistance;
            void *dataPtrs[4];
            dataPtrs[0] = (void *)(&smoothD);
            dataPtrs[1] = (void *)(&ini);
            dataPtrs[2] = (void *)(&fin);
            dataPtrs[3] = (void *)(&vPolys);
            lbfgs::lbfgs_parameter_t shortest_path_params;
            shortest_path_params.past = 3;
            shortest_path_params.delta = 1.0e-3;
            shortest_path_params.g_epsilon = 1.0e-5;

            lbfgs::lbfgs_optimize(xi,
                                  minDistance,
                                  &GCOPTER_PolytopeSFC_Flapping::costDistance,
                                  nullptr,
                                  nullptr,
                                  dataPtrs,
                                  shortest_path_params);

            path.resize(3, overlaps + 2);
            path.leftCols<1>() = ini;
            path.rightCols<1>() = fin;
            Eigen::VectorXd r;
            for (int i = 0, j = 0, k; i < overlaps; i++, j += k)
            {
                k = vPolys[2 * i + 1].cols();
                Eigen::Map<const Eigen::VectorXd> q(xi.data() + j, k);
                r = q.normalized().head(k - 1);
                path.col(i + 1) = vPolys[2 * i + 1].rightCols(k - 1) * r.cwiseProduct(r) +
                                  vPolys[2 * i + 1].col(0);
            }

            return;
        }

        static inline bool processCorridor(const PolyhedraH &hPs,
                                           PolyhedraV &vPs)
        {
            const int sizeCorridor = hPs.size() - 1;

            vPs.clear();
            vPs.reserve(2 * sizeCorridor + 1);

            int nv;
            PolyhedronH curIH;
            PolyhedronV curIV, curIOB;
            for (int i = 0; i < sizeCorridor; i++)
            {
                if (!geo_utils::enumerateVs(hPs[i], curIV))
                {
                    return false;
                }
                nv = curIV.cols();
                curIOB.resize(3, nv);
                curIOB.col(0) = curIV.col(0);
                curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
                vPs.push_back(curIOB);

                curIH.resize(hPs[i].rows() + hPs[i + 1].rows(), 4);
                curIH.topRows(hPs[i].rows()) = hPs[i];
                curIH.bottomRows(hPs[i + 1].rows()) = hPs[i + 1];
                if (!geo_utils::enumerateVs(curIH, curIV))
                {
                    return false;
                }
                nv = curIV.cols();
                curIOB.resize(3, nv);
                curIOB.col(0) = curIV.col(0);
                curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
                vPs.push_back(curIOB);
            }

            if (!geo_utils::enumerateVs(hPs.back(), curIV))
            {
                return false;
            }
            nv = curIV.cols();
            curIOB.resize(3, nv);
            curIOB.col(0) = curIV.col(0);
            curIOB.rightCols(nv - 1) = curIV.rightCols(nv - 1).colwise() - curIV.col(0);
            vPs.push_back(curIOB);

            return true;
        }

        static inline void setInitial(const Eigen::Matrix3Xd &path,
                                      const double &speed,
                                      const Eigen::VectorXi &intervalNs,
                                      Eigen::Matrix3Xd &innerPoints,
                                      Eigen::VectorXd &timeAlloc)
        {
            const int sizeM = intervalNs.size();
            const int sizeN = intervalNs.sum();
            innerPoints.resize(3, sizeN - 1);
            timeAlloc.resize(sizeN);

            Eigen::Vector3d a, b, c;
            for (int i = 0, j = 0, k = 0, l; i < sizeM; i++)
            {
                l = intervalNs(i);
                a = path.col(i);
                b = path.col(i + 1);
                c = (b - a) / l;
                timeAlloc.segment(j, l).setConstant(c.norm() / speed);
                j += l;
                for (int m = 0; m < l; m++)
                {
                    if (i > 0 || m > 0)
                    {
                        innerPoints.col(k++) = a + c * m;
                    }
                }
            }
        }

        inline void requireConfigured() const
        {
            if (!configured)
            {
                throw std::runtime_error(
                    "GCOPTER_PolytopeSFC_Flapping is not configured: call "
                    "setup() first");
            }
            return;
        }

    public:
        GCOPTER_PolytopeSFC_Flapping() = default;

        // Whole-trajectory callback, before MINCO's gradient propagation.
        // gradT is at FIXED coefficients; gradC is at FIXED durations. This
        // preserves MINCO's coefficient/time chain including phase history.
        inline void setAdditionalPenalty(AdditionalPenalty callback)
        {
            additionalPenalty = std::move(callback);
        }

        inline void clearAdditionalPenalty() { additionalPenalty = nullptr; }
        inline bool hasAdditionalPenalty() const { return bool(additionalPenalty); }
        inline Eigen::VectorXd lastDecision() const
        {
            if (lastOptimizedDecision.size() == 0)
                throw std::runtime_error("call optimize before lastDecision");
            return lastOptimizedDecision;
        }

    private:
        inline void attachAdditionalPenalty(
            const Eigen::VectorXd &ts, const Eigen::MatrixX3d &cs,
            double &cost, Eigen::VectorXd &gradT, Eigen::MatrixX3d &gradC,
            FlappingPenaltyReport &report) const
        {
            if (!additionalPenalty) return;
            Eigen::VectorXd dt;
            Eigen::MatrixX3d dc;
            double violation = 0.0;
            const double value = additionalPenalty(ts, cs, dt, dc, violation);
            if (!flapping::detail::isFinite(value) || value < 0.0 ||
                !flapping::detail::isFinite(violation) || violation < 0.0 ||
                dt.size() != ts.size() || dc.rows() != cs.rows() || dc.cols() != 3 ||
                !flapping::detail::isFinite(dt) || !flapping::detail::isFinite(dc))
                throw std::invalid_argument("additional penalty returned invalid cost/gradients/violation");
            cost += value;
            gradT += dt;
            gradC += dc;
            report.additionalCost = value;
            report.additionalMaxViola = violation;
            if (violation > report.maxViola)
            {
                report.maxViola = violation;
                report.worst = FlappingViolation::Additional;
                report.worstPiece = -1;
                report.worstLocalTime = 0.0;
            }
            report.cost = cost;
        }

    public:

        // The flapping configuration vectors, in the index layouts declared
        // above. setup() validates every length and value and throws
        // std::invalid_argument naming the offending entry, so a flapping
        // configuration can never be read with quadrotor indices.
        //
        // flappingAllocation and weightWingThrust are the left/right wing split
        // (l_eff, F_min, F_max) and its scalar weight. They are explicit
        // arguments with no default: the four per-side bounds are a separate
        // constraint set from the 11/7/5 vectors, and a compiled-in default
        // would let a caller silently inherit one vehicle's numbers.
        static inline bool validateConfig(const Eigen::VectorXd &flappingBounds,
                                          const Eigen::VectorXd &flappingWeights,
                                          const Eigen::VectorXd &flappingPhysical,
                                          const Eigen::VectorXd &flappingAllocation,
                                          const double weightWingThrust,
                                          std::string &reason)
        {
            FlappingConfig unused;
            return FlappingConfig::parse(flappingBounds, flappingWeights,
                                         flappingPhysical, flappingAllocation,
                                         weightWingThrust, unused, reason);
        }

        inline bool setup(const double &timeWeight,
                          const Eigen::Matrix3d &initialPVA,
                          const Eigen::Matrix3d &terminalPVA,
                          const PolyhedraH &safeCorridor,
                          const double &lengthPerPiece,
                          const double &smoothingFactor,
                          const int &integralResolution,
                          const Eigen::VectorXd &flappingBounds,
                          const Eigen::VectorXd &flappingWeights,
                          const Eigen::VectorXd &flappingPhysical,
                          const Eigen::VectorXd &flappingAllocation,
                          const double &weightWingThrust)
        {
            std::string reason;
            FlappingConfig parsed;
            if (!FlappingConfig::parse(flappingBounds, flappingWeights,
                                       flappingPhysical, flappingAllocation,
                                       weightWingThrust, parsed, reason))
            {
                throw std::invalid_argument("GCOPTER_PolytopeSFC_Flapping::setup: " +
                                            reason);
            }
            if (!(timeWeight >= 0.0))
            {
                throw std::invalid_argument(
                    "GCOPTER_PolytopeSFC_Flapping::setup: timeWeight must be >= 0");
            }
            if (!(lengthPerPiece > 0.0))
            {
                throw std::invalid_argument(
                    "GCOPTER_PolytopeSFC_Flapping::setup: lengthPerPiece must be > 0");
            }
            if (!(smoothingFactor > 0.0))
            {
                throw std::invalid_argument(
                    "GCOPTER_PolytopeSFC_Flapping::setup: smoothingFactor must be > 0");
            }
            if (integralResolution <= 0)
            {
                throw std::invalid_argument(
                    "GCOPTER_PolytopeSFC_Flapping::setup: integralResolution must be > 0");
            }
            if (safeCorridor.empty() || safeCorridor.size() < 2)
            {
                throw std::invalid_argument(
                    "GCOPTER_PolytopeSFC_Flapping::setup: safeCorridor needs at "
                    "least two half-space polytopes");
            }

            rho = timeWeight;
            headPVA = initialPVA;
            tailPVA = terminalPVA;

            hPolytopes = safeCorridor;
            for (size_t i = 0; i < hPolytopes.size(); i++)
            {
                const Eigen::ArrayXd norms =
                    hPolytopes[i].leftCols<3>().rowwise().norm();
                hPolytopes[i].array().colwise() /= norms;
            }
            if (!processCorridor(hPolytopes, vPolytopes))
            {
                return false;
            }

            polyN = hPolytopes.size();
            smoothEps = smoothingFactor;
            integralRes = integralResolution;
            flappingCfg = parsed;
            lastOptimizedDecision.resize(0);
            flatmap.reset(parsed.mass, parsed.gravity, parsed.inertia);
            allocSpeed = parsed.velMax * 3.0;

            getShortestPath(headPVA.col(0), tailPVA.col(0),
                            vPolytopes, smoothEps, shortPath);
            const Eigen::Matrix3Xd deltas = shortPath.rightCols(polyN) - shortPath.leftCols(polyN);
            pieceIdx = (deltas.colwise().norm() / lengthPerPiece).cast<int>().transpose();
            pieceIdx.array() += 1;
            pieceN = pieceIdx.sum();

            temporalDim = pieceN;
            spatialDim = 0;
            vPolyIdx.resize(pieceN - 1);
            hPolyIdx.resize(pieceN);
            for (int i = 0, j = 0, k; i < polyN; i++)
            {
                k = pieceIdx(i);
                for (int l = 0; l < k; l++, j++)
                {
                    if (l < k - 1)
                    {
                        vPolyIdx(j) = 2 * i;
                        spatialDim += vPolytopes[2 * i].cols();
                    }
                    else if (i < polyN - 1)
                    {
                        vPolyIdx(j) = 2 * i + 1;
                        spatialDim += vPolytopes[2 * i + 1].cols();
                    }
                    hPolyIdx(j) = i;
                }
            }

            minco.setConditions(headPVA, tailPVA, pieceN);

            points.resize(3, pieceN - 1);
            times.resize(pieceN);
            gradByPoints.resize(3, pieceN - 1);
            gradByTimes.resize(pieceN);
            partialGradByCoeffs.resize(6 * pieceN, 3);
            partialGradByTimes.resize(pieceN);

            configured = true;
            return true;
        }

        inline bool isConfigured() const { return configured; }
        inline const FlappingConfig &config() const
        {
            requireConfigured();
            return flappingCfg;
        }
        inline int getPieceNum() const { return pieceN; }
        inline int getTemporalDim() const { return temporalDim; }
        inline int getSpatialDim() const { return spatialDim; }
        inline int getDecisionDim() const { return temporalDim + spatialDim; }
        inline double getTimeWeight() const { return rho; }

        // The decision vector the optimizer starts from, so a caller can report
        // the pre-optimization violation level against the same start.
        inline Eigen::VectorXd initialDecision() const
        {
            requireConfigured();
            Eigen::VectorXd x(temporalDim + spatialDim);
            Eigen::Map<Eigen::VectorXd> tau(x.data(), temporalDim);
            Eigen::Map<Eigen::VectorXd> xi(x.data() + temporalDim, spatialDim);
            Eigen::Matrix3Xd pts;
            Eigen::VectorXd ts;
            setInitial(shortPath, allocSpeed, pieceIdx, pts, ts);
            backwardT(ts, tau);
            backwardP(pts, vPolyIdx, vPolytopes, xi);
            return x;
        }

        inline void trajectoryFromDecision(const Eigen::VectorXd &x,
                                           Trajectory<5> &traj)
        {
            requireConfigured();
            if (x.size() != temporalDim + spatialDim)
            {
                throw std::invalid_argument(
                    "trajectoryFromDecision: x must have getDecisionDim() entries");
            }
            Eigen::Map<const Eigen::VectorXd> tau(x.data(), temporalDim);
            Eigen::Map<const Eigen::VectorXd> xi(x.data() + temporalDim, spatialDim);
            forwardT(tau, times);
            forwardP(xi, vPolyIdx, vPolytopes, points);
            minco.setParameters(points, times);
            minco.getTrajectory(traj);
        }

        // -------------------------------------------------------------------
        // Constraint cost and its exact gradients, as a pure function of a
        // (times, coeffs) pair. `coeffs` uses the MINCO layout: row 6i + k of
        // piece i holds the coefficient of s^k. The pair does not have to come
        // from a feasible MINCO solve, which is what makes this usable as a
        // finite-difference target.
        // -------------------------------------------------------------------
        inline void evaluatePenalty(const Eigen::VectorXd &timesIn,
                                    const Eigen::MatrixX3d &coeffsIn,
                                    double &cost,
                                    Eigen::VectorXd &gradT,
                                    Eigen::MatrixX3d &gradC,
                                    FlappingPenaltyReport &report) const
        {
            requireConfigured();
            if (timesIn.size() != pieceN)
            {
                throw std::invalid_argument(
                    "evaluatePenalty: times must have pieceN entries");
            }
            if (coeffsIn.rows() != 6 * pieceN || coeffsIn.cols() != 3)
            {
                throw std::invalid_argument(
                    "evaluatePenalty: coeffs must be (6 * pieceN) x 3");
            }
            cost = 0.0;
            gradT.setZero(timesIn.size());
            gradC.setZero(coeffsIn.rows(), 3);
            report = FlappingPenaltyReport();
            attachPenaltyFunctional(timesIn, coeffsIn, hPolyIdx, hPolytopes,
                                    smoothEps, integralRes, flappingCfg, flatmap,
                                    cost, gradT, gradC, &report);
            attachAdditionalPenalty(timesIn, coeffsIn, cost, gradT, gradC, report);
        }

    private:
        // Piece<5>::getCoeffMat() stores the coefficients in descending order
        // (col k multiplies s^(5-k)), while the MINCO layout used here stores
        // row 6i + k as the coefficient of s^k. The reversal is the whole point
        // of this helper: getting it wrong silently swaps snap and position.
        static inline Eigen::Matrix<double, 6, 3> trajectoryCoeffs(
            const Piece<5> &piece)
        {
            return piece.getCoeffMat().transpose().colwise().reverse();
        }

    public:
        // The (times, coeffs) pair a trajectory with setup()'s piece count
        // decodes to, in the layout evaluatePenalty() consumes. reportViolations()
        // and the pybind binding both go through here, so the coefficient
        // reordering lives in exactly one place.
        inline void trajectoryToFlat(const Trajectory<5> &traj,
                                     Eigen::VectorXd &timesOut,
                                     Eigen::MatrixX3d &coeffsOut) const
        {
            requireConfigured();
            if (traj.getPieceNum() != pieceN)
            {
                throw std::invalid_argument(
                    "trajectoryToFlat: trajectory piece count does not match setup()");
            }
            timesOut.resize(pieceN);
            coeffsOut.resize(6 * pieceN, 3);
            for (int i = 0; i < pieceN; i++)
            {
                timesOut(i) = traj[i].getDuration();
                coeffsOut.block<6, 3>(6 * i, 0) = trajectoryCoeffs(traj[i]);
            }
        }

        // Constraint report for an arbitrary trajectory, no gradients.
        inline FlappingPenaltyReport reportViolations(const Trajectory<5> &traj) const
        {
            Eigen::VectorXd ts;
            Eigen::MatrixX3d coeffs;
            trajectoryToFlat(traj, ts, coeffs);
            double cost;
            Eigen::VectorXd gradT;
            Eigen::MatrixX3d gradC;
            FlappingPenaltyReport report;
            evaluatePenalty(ts, coeffs, cost, gradT, gradC, report);
            return report;
        }

        // Flat form of the trajectory the decision vector x decodes to, in the
        // same layout evaluatePenalty() and evaluateObjective() consume. Exposed
        // so a caller can finite-difference the penalty directly on the MINCO
        // coefficients and durations.
        inline void decisionToFlat(const Eigen::VectorXd &x,
                                   Eigen::VectorXd &timesOut,
                                   Eigen::MatrixX3d &coeffsOut)
        {
            requireConfigured();
            if (x.size() != temporalDim + spatialDim)
            {
                throw std::invalid_argument(
                    "decisionToFlat: x must have getDecisionDim() entries");
            }
            Eigen::Map<const Eigen::VectorXd> tau(x.data(), temporalDim);
            Eigen::Map<const Eigen::VectorXd> xi(x.data() + temporalDim, spatialDim);
            forwardT(tau, times);
            forwardP(xi, vPolyIdx, vPolytopes, points);
            minco.setParameters(points, times);
            timesOut = times;
            coeffsOut = minco.getCoeffs();
        }

        // -------------------------------------------------------------------
        // The full objective the L-BFGS loop minimizes:
        //     MINCO energy + timeWeight * sum(times) + flapping penalty,
        // with the gradient with respect to the decision vector x. This is the
        // same code path optimize() runs, exposed so a finite-difference check
        // exercises the real objective rather than a copy of it.
        // -------------------------------------------------------------------
        inline double evaluateObjective(const Eigen::VectorXd &x,
                                        Eigen::VectorXd &g,
                                        FlappingPenaltyReport &report)
        {
            requireConfigured();
            if (x.size() != temporalDim + spatialDim)
            {
                throw std::invalid_argument(
                    "evaluateObjective: x must have getDecisionDim() entries");
            }
            if (g.size() != x.size())
            {
                throw std::invalid_argument(
                    "evaluateObjective: g must have the same size as x");
            }

            const double weightT = rho;
            Eigen::Map<const Eigen::VectorXd> tau(x.data(), temporalDim);
            Eigen::Map<const Eigen::VectorXd> xi(x.data() + temporalDim, spatialDim);
            Eigen::Map<Eigen::VectorXd> gradTau(g.data(), temporalDim);
            Eigen::Map<Eigen::VectorXd> gradXi(g.data() + temporalDim, spatialDim);

            forwardT(tau, times);
            forwardP(xi, vPolyIdx, vPolytopes, points);

            double cost;
            minco.setParameters(points, times);
            minco.getEnergy(cost);
            minco.getEnergyPartialGradByCoeffs(partialGradByCoeffs);
            minco.getEnergyPartialGradByTimes(partialGradByTimes);

            attachPenaltyFunctional(times, minco.getCoeffs(),
                                    hPolyIdx, hPolytopes,
                                    smoothEps, integralRes,
                                    flappingCfg, flatmap,
                                    cost, partialGradByTimes, partialGradByCoeffs,
                                    &report);

            attachAdditionalPenalty(times, minco.getCoeffs(), cost,
                                    partialGradByTimes, partialGradByCoeffs, report);

            minco.propogateGrad(partialGradByCoeffs, partialGradByTimes,
                                gradByPoints, gradByTimes);

            cost += weightT * times.sum();
            gradByTimes.array() += weightT;

            backwardGradT(tau, gradByTimes, gradTau);
            backwardGradP(xi, vPolyIdx, vPolytopes, gradByPoints, gradXi);
            normRetrictionLayer(xi, vPolyIdx, vPolytopes, cost, gradXi);

            report.cost = cost;
            return cost;
        }

        inline double evaluateObjective(const Eigen::VectorXd &x,
                                        Eigen::VectorXd &g)
        {
            FlappingPenaltyReport ignored;
            return evaluateObjective(x, g, ignored);
        }

        inline double optimize(Trajectory<5> &traj,
                               const double &relCostTol)
        {
            requireConfigured();
            Eigen::VectorXd x(temporalDim + spatialDim);
            Eigen::Map<Eigen::VectorXd> tau(x.data(), temporalDim);
            Eigen::Map<Eigen::VectorXd> xi(x.data() + temporalDim, spatialDim);

            setInitial(shortPath, allocSpeed, pieceIdx, points, times);
            backwardT(times, tau);
            backwardP(points, vPolyIdx, vPolytopes, xi);

            double minCostFunctional;
            lbfgs_params.mem_size = 256;
            lbfgs_params.past = 3;
            lbfgs_params.min_step = 1.0e-32;
            lbfgs_params.g_epsilon = 0.0;
            lbfgs_params.delta = relCostTol;

            int ret = lbfgs::lbfgs_optimize(x,
                                            minCostFunctional,
                                            &GCOPTER_PolytopeSFC_Flapping::costFunctional,
                                            nullptr,
                                            nullptr,
                                            this,
                                            lbfgs_params);

            lastOptimizedDecision = x;

            if (ret >= 0)
            {
                forwardT(tau, times);
                forwardP(xi, vPolyIdx, vPolytopes, points);
                minco.setParameters(points, times);
                minco.getTrajectory(traj);
            }
            else
            {
                traj.clear();
                minCostFunctional = INFINITY;
                std::cout << "Optimization Failed: "
                          << lbfgs::lbfgs_strerror(ret)
                          << std::endl;
            }

            return minCostFunctional;
        }
    };

}

#endif
