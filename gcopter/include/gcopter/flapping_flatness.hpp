/*
    Cycle-averaged flapping-wing differential flatness map: simplified
    total-thrust / total-moment model (paper equation (40)).

    The wingbeat cycle is averaged away entirely, so the vehicle carries a
    single thrust magnitude T along a single force axis z and a single total
    body moment. There is no left/right wing mixer, no force-axis cant beta_f,
    no k_T f^2 frequency law and no translational or rotational drag. The only
    parameters are (m, J, g): mass, body-frame inertia diagonal and gravity
    magnitude. That earlier, richer prototype is gone from this header; it is
    preserved by scripts/gcopter_flapping_flatness.patch, which is kept as a
    legacy record only and is no longer used to build anything.

    Independent companion to flatness::FlatnessMap (flatness.hpp). The
    quadrotor map is left untouched; this header adds a second, self-contained
    map for the flapping vehicle.

    Coordinate and sign conventions (identical to flatness.hpp):

      * q_wb = [w, x, y, z] (Hamilton, scalar first) rotates body-frame vectors
        into the world frame:  v_w = R(q_wb) v_b.
      * world is z-up:  g_w = [0, 0, -g], so u = a - g_w = a + [0, 0, g].
      * body +Z is the cycle-averaged force axis.
      * the attitude uses the same tilt (x) yaw construction as
        flatness::FlatnessMap:  q = q_tilt(z) (x) q_yaw(psi), with
        q_tilt = [0.5 d, -z_y / d, z_x / d, 0],  d = sqrt(2 (1 + z_z)),
        q_yaw  = [cos(psi/2), 0, 0, sin(psi/2)].

    Forward mapping:

      u    = a + [0, 0, g]                       (specific force, m/s^2)
      T    = m ||u||                             (total thrust, N)
      z    = u / ||u||                           (force axis, world frame)
      Zb   = m jerk,  Zbb = m snap               (force derivative, N/s, N/s^2)
      Td   = z . Zb
      Tdd  = (Zb.Zb) / T + z.Zbb - Td^2 / T
      zd   = (Zb - Td z) / T
      zdd  = (Zbb - 2 (Td/T) Zb - Tdd z + 2 (Td/T) Td z) / T

      q    = q_tilt(z) (x) q_yaw(psi)
      omega     = 2 vec(q* qd)     (body rate, rad/s)
      omega_dot = 2 vec(q* qdd)    (body angular acceleration, rad/s^2)
      Gamma     = J omega_dot + omega x (J omega)      (N m)

    With zero drag and beta_f = 0 the quadrotor map's (thrust, quat, omega) are
    reproduced exactly, because flatness::FlatnessMap computes the same tilt
    quaternion, the same z direction and thr = z . (m a + m g_w) = m ||u||. That
    regression is covered by tests/test_flapping_flatness.py.

    Analytic gradients: backward() is a hand-derived reverse-mode pass over the
    mapping above. It was validated against central differences of forward()
    before being ported, and tests/test_flapping_flatness.py revalidates the
    compiled version. It differentiates THIS map only; it says nothing about
    GCOPTER costs or penalties, which are not implemented here (see below).

    Non-finite inputs are a caller error, not an infeasible state: forward()
    and backward() throw std::invalid_argument for NaN/Inf arguments instead of
    reporting ThrustBelowEpsilon. The tests use detail::isFinite(), a bit-level
    check, because the canonical -Ofast build would otherwise fold
    std::isfinite() and Eigen's allFinite() to `true`.

    NOT implemented here (do not treat as done):

      * GCOPTER feasibility penalties and costs, and the collected
        flat-output -> constraint mapping they would need.
      * Any flapping plant / actuator / mixer model. This map produces T and a
        total moment only; it never speaks of wings, frequencies or surfaces.

    Units: SI, angles in radians.
*/

#ifndef FLAPPING_FLATNESS_HPP
#define FLAPPING_FLATNESS_HPP

#include <Eigen/Eigen>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace flapping
{
    namespace detail
    {
        // NaN/Inf test that survives -ffast-math.
        //
        // The canonical gcopter_planner target is compiled with -Ofast, which
        // implies -ffast-math: the compiler may then assume no NaN or Inf exists,
        // so std::isfinite() folds to `true` and Eigen's allFinite() (built from
        // `x == x`) becomes unreliable. Both were observed to accept Inf thrust
        // and NaN parameters in the gcopter_planner build. Testing the exponent
        // bits uses integer operations, which fast-math does not touch.
        inline bool isFinite(const double value)
        {
            constexpr std::uint64_t kExponentMask = 0x7ff0000000000000ULL;
            std::uint64_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            return (bits & kExponentMask) != kExponentMask;
        }

        template <typename Derived>
        inline bool isFinite(const Eigen::DenseBase<Derived> &value)
        {
            for (Eigen::Index i = 0; i < value.size(); ++i)
            {
                if (!isFinite(value.derived().data()[i]))
                {
                    return false;
                }
            }
            return true;
        }

        // Hamilton product, scalar first: returns a (x) b.
        inline Eigen::Vector4d quatMul(const Eigen::Vector4d &a,
                                       const Eigen::Vector4d &b)
        {
            return Eigen::Vector4d(
                a(0) * b(0) - a(1) * b(1) - a(2) * b(2) - a(3) * b(3),
                a(0) * b(1) + a(1) * b(0) + a(2) * b(3) - a(3) * b(2),
                a(0) * b(2) - a(1) * b(3) + a(2) * b(0) + a(3) * b(1),
                a(0) * b(3) + a(1) * b(2) - a(2) * b(1) + a(3) * b(0));
        }

        inline Eigen::Vector4d quatConj(const Eigen::Vector4d &a)
        {
            return Eigen::Vector4d(a(0), -a(1), -a(2), -a(3));
        }

        // v_world = R(q) v_body.
        inline Eigen::Matrix3d quatToRotation(const Eigen::Vector4d &q)
        {
            const double w = q(0), x = q(1), y = q(2), z = q(3);
            Eigen::Matrix3d R;
            R << 1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z),
                2.0 * (x * z + w * y),
                2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - w * x),
                2.0 * (x * z - w * y), 2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y);
            return R;
        }
    }  // namespace detail

    // Outcome of forward() and backward().
    enum class ForwardStatus
    {
        Ok = 0,
        // T = m ||a - g_w|| is at or below the numerical epsilon, so the force
        // direction is undefined and no attitude can be built.
        ThrustBelowEpsilon = 1,
        // The force axis points (near) straight down: the tilt/yaw construction
        // inverts d = sqrt(2 (1 + z_z)). quat, R, omega, omega_dot and gamma are
        // NaN in this case.
        AttitudeSingular = 2,
    };

    inline const char *toString(const ForwardStatus status)
    {
        switch (status)
        {
        case ForwardStatus::Ok:
            return "Ok";
        case ForwardStatus::ThrustBelowEpsilon:
            return "ThrustBelowEpsilon";
        case ForwardStatus::AttitudeSingular:
            return "AttitudeSingular";
        }
        return "Unknown";
    }

    // The whole parameter set of the simplified model.
    struct FlappingParams
    {
        // Mass [kg], strictly positive.
        double m = std::numeric_limits<double>::quiet_NaN();
        // Body-frame inertia diagonal [kg m^2], strictly positive.
        Eigen::Vector3d J_diag = Eigen::Vector3d::Constant(
            std::numeric_limits<double>::quiet_NaN());
        // Gravitational acceleration magnitude [m/s^2], non-negative.
        double g = std::numeric_limits<double>::quiet_NaN();

        // True once the map has a usable (m, J, g): all finite, m > 0, J > 0,
        // g >= 0. A non-positive mass or inertia would make the map meaningless
        // rather than merely inaccurate, so it is rejected here.
        bool isComplete() const
        {
            return detail::isFinite(m) && detail::isFinite(J_diag) &&
                   detail::isFinite(g) && m > 0.0 && J_diag(0) > 0.0 &&
                   J_diag(1) > 0.0 && J_diag(2) > 0.0 && g >= 0.0;
        }
    };

    // Output of forward().
    struct ForwardResult
    {
        ForwardStatus status = ForwardStatus::Ok;
        // Total cycle-averaged thrust T = m ||a - g_w|| [N].
        double thrust = std::numeric_limits<double>::quiet_NaN();
        // Unit force axis z = (a - g_w) / ||a - g_w||, world frame.
        Eigen::Vector3d force_dir = Eigen::Vector3d::Constant(
            std::numeric_limits<double>::quiet_NaN());
        // dz/dt implied by the jerk [1/s], world frame. Exposed because the
        // structure z . zd = 0 is a cheap correctness probe.
        Eigen::Vector3d force_dir_dot = Eigen::Vector3d::Constant(
            std::numeric_limits<double>::quiet_NaN());
        // d^2 z/dt^2 implied by the snap [1/s^2], world frame. Satisfies
        // z . zdd + ||zd||^2 = 0.
        Eigen::Vector3d force_dir_ddot = Eigen::Vector3d::Constant(
            std::numeric_limits<double>::quiet_NaN());
        // Attitude q_wb = [w, x, y, z]; NaN when status != Ok.
        Eigen::Vector4d quat = Eigen::Vector4d::Constant(
            std::numeric_limits<double>::quiet_NaN());
        // R(q_wb), world <- body; NaN when status != Ok.
        Eigen::Matrix3d R = Eigen::Matrix3d::Constant(
            std::numeric_limits<double>::quiet_NaN());
        // Body angular velocity omega_d [rad/s] from the jerk; NaN when
        // status != Ok.
        Eigen::Vector3d omega = Eigen::Vector3d::Constant(
            std::numeric_limits<double>::quiet_NaN());
        // Body angular acceleration omega_dot_d [rad/s^2] from the snap; NaN
        // when status != Ok.
        Eigen::Vector3d omega_dot = Eigen::Vector3d::Constant(
            std::numeric_limits<double>::quiet_NaN());
        // Required total moment Gamma = J omega_dot + omega x (J omega) [N m];
        // NaN when status != Ok.
        Eigen::Vector3d gamma = Eigen::Vector3d::Constant(
            std::numeric_limits<double>::quiet_NaN());

        bool ok() const { return status == ForwardStatus::Ok; }
    };

    // Output of backward(): the gradient of a scalar cost L with respect to the
    // flat inputs, given the upstream gradients dL/d(thrust, quat, omega,
    // omega_dot, gamma).
    struct BackwardResult
    {
        // dL/d(vel) is identically zero: the map never reads the velocity. Kept
        // so the gradient tuple has the same shape as the input tuple.
        Eigen::Vector3d vel = Eigen::Vector3d::Zero();
        Eigen::Vector3d acc = Eigen::Vector3d::Constant(
            std::numeric_limits<double>::quiet_NaN());
        Eigen::Vector3d jer = Eigen::Vector3d::Constant(
            std::numeric_limits<double>::quiet_NaN());
        Eigen::Vector3d snap = Eigen::Vector3d::Constant(
            std::numeric_limits<double>::quiet_NaN());
        double psi = std::numeric_limits<double>::quiet_NaN();
        double dpsi = std::numeric_limits<double>::quiet_NaN();
        double ddpsi = std::numeric_limits<double>::quiet_NaN();

        bool ok() const
        {
            return detail::isFinite(acc) && detail::isFinite(jer) &&
                   detail::isFinite(snap) && detail::isFinite(psi) &&
                   detail::isFinite(dpsi) && detail::isFinite(ddpsi);
        }
    };

    class FlappingFlatnessMap
    {
    public:
        FlappingFlatnessMap() = default;

        // Numerical guard for degenerate thrust, [N]. Not a physical parameter:
        // it only decides when the force direction stops being usable.
        inline void setThrustEpsilon(const double eps)
        {
            if (!(eps >= 0.0))
            {
                throw std::invalid_argument("thrust epsilon must be non-negative");
            }
            thrust_eps = eps;
            return;
        }

        inline double getThrustEpsilon() const { return thrust_eps; }

        inline void reset(const FlappingParams &params)
        {
            if (!params.isComplete())
            {
                throw std::invalid_argument(
                    "FlappingParams is incomplete: m and g must be finite with "
                    "m > 0, g >= 0, and J_diag must be finite and strictly "
                    "positive");
            }
            params_ = params;
            return;
        }

        inline void reset(const double vehicle_mass,
                          const double gravitational_acceleration,
                          const Eigen::Vector3d &inertia_diag)
        {
            FlappingParams p;
            p.m = vehicle_mass;
            p.g = gravitational_acceleration;
            p.J_diag = inertia_diag;
            reset(p);
            return;
        }

        inline bool isConfigured() const { return params_.isComplete(); }

        inline const FlappingParams &params() const { return params_; }

        // ------------------------------------------------------------------
        // Forward: (vel, acc, jerk, snap, psi, dpsi, ddpsi)
        //          -> (T, z, zd, zdd, quat, R, omega, omega_dot, gamma)
        // ------------------------------------------------------------------
        // vel is accepted but not read: the averaged model has no velocity
        // dependence. It is kept so the signature and the backward gradient
        // tuple line up with the flat-output ordering GCOPTER optimises.
        inline ForwardResult forward(const Eigen::Vector3d &vel,
                                     const Eigen::Vector3d &acc,
                                     const Eigen::Vector3d &jer,
                                     const Eigen::Vector3d &snap,
                                     const double &psi,
                                     const double &dpsi = 0.0,
                                     const double &ddpsi = 0.0) const
        {
            requireConfigured();

            // A non-finite input is a caller error, not an infeasible state: it
            // would otherwise fall through the `T > thrust_eps` test as
            // ThrustBelowEpsilon and be reported as a zero-thrust command.
            if (!detail::isFinite(vel) || !detail::isFinite(acc) ||
                !detail::isFinite(jer) || !detail::isFinite(snap) ||
                !detail::isFinite(psi) || !detail::isFinite(dpsi) ||
                !detail::isFinite(ddpsi))
            {
                throw std::invalid_argument(
                    "forward: vel, acc, jer, snap, psi, dpsi and ddpsi must be "
                    "finite");
            }

            ForwardResult result;
            Cache cache;
            result.status = evaluate(params_, thrust_eps, acc, jer, snap, psi,
                                     dpsi, ddpsi, result, cache);
            return result;
        }

        // ------------------------------------------------------------------
        // Backward: reverse-mode gradient of a scalar cost w.r.t. the inputs.
        // ------------------------------------------------------------------
        // Every gradient argument defaults to zero, so a bare call returns an
        // all-zero (still finite) result rather than a silently wrong one.
        //
        // The pass differentiates this mapping only. It does not know about
        // GCOPTER costs, penalties or the collector; whoever builds those must
        // chain their own gradients into these upstream arguments.
        //
        // Returns NaN everywhere when forward() is not Ok (a non-differentiable
        // point), and throws std::invalid_argument on non-finite arguments.
        inline BackwardResult backward(
            const Eigen::Vector3d &vel, const Eigen::Vector3d &acc,
            const Eigen::Vector3d &jer, const Eigen::Vector3d &snap,
            const double &psi, const double &dpsi = 0.0,
            const double &ddpsi = 0.0, const double &thrust_grad = 0.0,
            const Eigen::Vector4d &quat_grad = Eigen::Vector4d::Zero(),
            const Eigen::Vector3d &omega_grad = Eigen::Vector3d::Zero(),
            const Eigen::Vector3d &omega_dot_grad = Eigen::Vector3d::Zero(),
            const Eigen::Vector3d &gamma_grad = Eigen::Vector3d::Zero()) const
        {
            requireConfigured();

            if (!detail::isFinite(vel) || !detail::isFinite(acc) ||
                !detail::isFinite(jer) || !detail::isFinite(snap) ||
                !detail::isFinite(psi) || !detail::isFinite(dpsi) ||
                !detail::isFinite(ddpsi) || !detail::isFinite(thrust_grad) ||
                !detail::isFinite(quat_grad) || !detail::isFinite(omega_grad) ||
                !detail::isFinite(omega_dot_grad) ||
                !detail::isFinite(gamma_grad))
            {
                throw std::invalid_argument(
                    "backward: every argument must be finite");
            }

            ForwardResult ignored;
            Cache cache;
            const ForwardStatus status =
                evaluate(params_, thrust_eps, acc, jer, snap, psi, dpsi, ddpsi,
                         ignored, cache);
            if (status != ForwardStatus::Ok)
            {
                BackwardResult failed;
                failed.vel = Eigen::Vector3d::Constant(
                    std::numeric_limits<double>::quiet_NaN());
                return failed;
            }

            return sweep(params_, cache, thrust_grad, quat_grad, omega_grad,
                         omega_dot_grad, gamma_grad);
        }

    private:
        struct Cache
        {
            Eigen::Vector3d u, z, zd, zdd, Zb, Zbb;
            double r = 0.0, T = 0.0, Td = 0.0, Tdd = 0.0;
            double v0 = 0.0, u1 = 0.0;
            double d = 1.0, lam0 = 0.0, lam1 = 0.0, lam3 = 0.0;
            double t0 = 0.0, t1 = 0.0, t2 = 0.0;
            double ch = 0.0, sh = 0.0, psi = 0.0, dpsi = 0.0, ddpsi = 0.0;
            Eigen::Vector4d qt, qtd, qtdd, qy, qyd, qydd, q, qd, qdd;
            Eigen::Vector3d omg, omgd, Jw;
        };

        static constexpr double kSingularEpsilon = 1e-8;

        inline void requireConfigured() const
        {
            if (!isConfigured())
            {
                throw std::runtime_error(
                    "FlappingFlatnessMap is not configured: call reset(m, g, "
                    "J_diag) first");
            }
            return;
        }

        // Shared by forward() and backward(): fills `result` and the cache.
        // The caller has already checked the arguments for finiteness.
        static ForwardStatus evaluate(const FlappingParams &p,
                                      const double thrust_eps,
                                      const Eigen::Vector3d &acc,
                                      const Eigen::Vector3d &jer,
                                      const Eigen::Vector3d &snap,
                                      const double psi, const double dpsi,
                                      const double ddpsi,
                                      ForwardResult &result, Cache &c)
        {
            c.psi = psi;
            c.dpsi = dpsi;
            c.ddpsi = ddpsi;

            // u = a - g_w with g_w = [0, 0, -g].
            const Eigen::Vector3d u = acc + Eigen::Vector3d(0.0, 0.0, p.g);
            const double r = u.norm();
            const double T = p.m * r;
            c.u = u;
            c.r = r;
            c.T = T;
            result.thrust = T;
            if (!(T > thrust_eps))
            {
                return ForwardStatus::ThrustBelowEpsilon;
            }

            const Eigen::Vector3d z = u / r;
            c.z = z;
            result.force_dir = z;

            c.Zb = p.m * jer;
            c.Zbb = p.m * snap;
            c.Td = z.dot(c.Zb);
            c.Tdd = c.Zb.dot(c.Zb) / T + z.dot(c.Zbb) - c.Td * c.Td / T;
            c.v0 = 1.0 / T;
            c.u1 = c.Td * c.v0;
            c.zd = c.v0 * (c.Zb - c.Td * z);
            c.zdd =
                c.v0 * (c.Zbb - 2.0 * c.u1 * c.Zb - c.Tdd * z + 2.0 * c.u1 * c.Td * z);
            result.force_dir_dot = c.zd;
            result.force_dir_ddot = c.zdd;

            // The tilt/yaw construction divides by d = sqrt(2 (1 + z_z)) and
            // the rate formula by (1 + z_z): both vanish when the required
            // force points straight down, where no yaw-referenced attitude
            // exists.
            const double omg_den = z.z() + 1.0;
            if (!(omg_den > kSingularEpsilon))
            {
                return ForwardStatus::AttitudeSingular;
            }

            const double d = std::sqrt(2.0 * (1.0 + z.z()));
            const double lam0 = 1.0 / d;
            const double lam1 = 1.0 / (d * d * d);
            const double lam3 = 1.0 / (d * d * d * d * d);
            c.d = d;
            c.lam0 = lam0;
            c.lam1 = lam1;
            c.lam3 = lam3;

            c.t0 = 0.5 * d;
            c.t1 = -z.y() / d;
            c.t2 = z.x() / d;
            const Eigen::Vector3d &zd = c.zd;
            const Eigen::Vector3d &zdd = c.zdd;

            c.qt = Eigen::Vector4d(c.t0, c.t1, c.t2, 0.0);
            c.qtd = Eigen::Vector4d(
                0.5 * lam0 * zd(2), -lam0 * zd(1) + z.y() * lam1 * zd(2),
                lam0 * zd(0) - z.x() * lam1 * zd(2), 0.0);
            c.qtdd = Eigen::Vector4d(
                0.5 * lam0 * zdd(2) - 0.5 * lam1 * zd(2) * zd(2),
                -lam0 * zdd(1) + z.y() * lam1 * zdd(2) +
                    2.0 * lam1 * zd(1) * zd(2) -
                    3.0 * z.y() * lam3 * zd(2) * zd(2),
                lam0 * zdd(0) - z.x() * lam1 * zdd(2) -
                    2.0 * lam1 * zd(0) * zd(2) +
                    3.0 * z.x() * lam3 * zd(2) * zd(2),
                0.0);

            c.ch = std::cos(0.5 * psi);
            c.sh = std::sin(0.5 * psi);
            c.qy = Eigen::Vector4d(c.ch, 0.0, 0.0, c.sh);
            c.qyd = 0.5 * dpsi * Eigen::Vector4d(-c.sh, 0.0, 0.0, c.ch);
            c.qydd = 0.5 * ddpsi * Eigen::Vector4d(-c.sh, 0.0, 0.0, c.ch) +
                     0.25 * dpsi * dpsi * Eigen::Vector4d(-c.ch, 0.0, 0.0, -c.sh);

            c.q = detail::quatMul(c.qt, c.qy);
            c.qd = detail::quatMul(c.qtd, c.qy) + detail::quatMul(c.qt, c.qyd);
            c.qdd = detail::quatMul(c.qtdd, c.qy) +
                    2.0 * detail::quatMul(c.qtd, c.qyd) +
                    detail::quatMul(c.qt, c.qydd);

            // omega = 2 vec(q* qd), omega_dot = 2 vec(q* qdd); q is unit, so
            // the q* q term drops out of the derivative of the first.
            Eigen::Vector4d omg_q = detail::quatMul(detail::quatConj(c.q), c.qd);
            c.omg = 2.0 * omg_q.tail<3>();
            Eigen::Vector4d omgd_q = detail::quatMul(detail::quatConj(c.q), c.qdd);
            c.omgd = 2.0 * omgd_q.tail<3>();
            c.Jw = p.J_diag.cwiseProduct(c.omg);

            result.quat = c.q;
            result.R = detail::quatToRotation(c.q);
            result.omega = c.omg;
            result.omega_dot = c.omgd;
            result.gamma = p.J_diag.cwiseProduct(c.omgd) + c.omg.cross(c.Jw);
            return ForwardStatus::Ok;
        }

        // Hand-derived reverse mode. The forward expressions are repeated here
        // in the same order they appear in evaluate(); each line's comment
        // names the forward line it differentiates, so the two stay auditable
        // side by side. `_b` variables hold dL/d(forward variable).
        static BackwardResult sweep(const FlappingParams &p, const Cache &c,
                                    const double thrust_grad,
                                    const Eigen::Vector4d &quat_grad,
                                    const Eigen::Vector3d &omega_grad,
                                    const Eigen::Vector3d &omega_dot_grad,
                                    const Eigen::Vector3d &gamma_grad)
        {
            const Eigen::Vector3d &z = c.z;
            const Eigen::Vector3d &zd = c.zd;
            const Eigen::Vector3d &zdd = c.zdd;
            const Eigen::Vector3d &J = p.J_diag;

            Eigen::Vector3d z_b = Eigen::Vector3d::Zero();
            Eigen::Vector3d zd_b = Eigen::Vector3d::Zero();
            Eigen::Vector3d zdd_b = Eigen::Vector3d::Zero();

            // Gamma = J omgd + omg x (J omg).
            Eigen::Vector3d omgd_b = J.cwiseProduct(gamma_grad);
            const Eigen::Vector3d Jw_b = gamma_grad.cross(c.omg);
            Eigen::Vector3d omg_b =
                c.Jw.cross(gamma_grad) + J.cwiseProduct(Jw_b) + omega_grad;
            omgd_b += omega_dot_grad;

            // omg = 2 vec(q* (x) qd); omgd = 2 vec(q* (x) qdd).
            // Adjoint of the Hamilton product: for r = a (x) b, r_b -> a_b gets
            // r_b (x) b*, b_b gets a* (x) r_b.
            Eigen::Vector4d q_b = Eigen::Vector4d::Zero();
            Eigen::Vector4d qd_b = Eigen::Vector4d::Zero();
            Eigen::Vector4d qdd_b = Eigen::Vector4d::Zero();

            Eigen::Vector4d g1 = Eigen::Vector4d::Zero();
            g1.tail<3>() = 2.0 * omg_b;
            const Eigen::Vector4d qs_b =
                detail::quatMul(g1, detail::quatConj(c.qd));
            qd_b += detail::quatMul(c.q, g1);
            q_b += detail::quatConj(qs_b);

            Eigen::Vector4d g2 = Eigen::Vector4d::Zero();
            g2.tail<3>() = 2.0 * omgd_b;
            const Eigen::Vector4d qs_b2 =
                detail::quatMul(g2, detail::quatConj(c.qdd));
            qdd_b += detail::quatMul(c.q, g2);
            q_b += detail::quatConj(qs_b2);
            q_b += quat_grad;

            // q = qt (x) qy
            // qd = qtd (x) qy + qt (x) qyd
            // qdd = qtdd (x) qy + 2 qtd (x) qyd + qt (x) qydd
            Eigen::Vector4d qt_b = Eigen::Vector4d::Zero();
            Eigen::Vector4d qy_b = Eigen::Vector4d::Zero();
            Eigen::Vector4d qtd_b = Eigen::Vector4d::Zero();
            Eigen::Vector4d qyd_b = Eigen::Vector4d::Zero();
            Eigen::Vector4d qtdd_b = Eigen::Vector4d::Zero();
            Eigen::Vector4d qydd_b = Eigen::Vector4d::Zero();

            qt_b += detail::quatMul(q_b, detail::quatConj(c.qy));
            qy_b += detail::quatMul(detail::quatConj(c.qt), q_b);
            qtd_b += detail::quatMul(qd_b, detail::quatConj(c.qy));
            qy_b += detail::quatMul(detail::quatConj(c.qtd), qd_b);
            qt_b += detail::quatMul(qd_b, detail::quatConj(c.qyd));
            qyd_b += detail::quatMul(detail::quatConj(c.qt), qd_b);
            qtdd_b += detail::quatMul(qdd_b, detail::quatConj(c.qy));
            qy_b += detail::quatMul(detail::quatConj(c.qtdd), qdd_b);
            qtd_b += 2.0 * detail::quatMul(qdd_b, detail::quatConj(c.qyd));
            qyd_b += 2.0 * detail::quatMul(detail::quatConj(c.qtd), qdd_b);
            qt_b += detail::quatMul(qdd_b, detail::quatConj(c.qydd));
            qydd_b += detail::quatMul(detail::quatConj(c.qt), qdd_b);

            // qy   = [ch, 0, 0, sh]
            // qyd  = 0.5 dpsi a1,          a1 = [-sh, 0, 0, ch]
            // qydd = 0.5 ddpsi a1 + 0.25 dpsi^2 a2,  a2 = [-ch, 0, 0, -sh]
            // da1/dpsi = 0.5 a2, da2/dpsi = 0.5 a1
            const Eigen::Vector4d a1(-c.sh, 0.0, 0.0, c.ch);
            const Eigen::Vector4d a2(-c.ch, 0.0, 0.0, -c.sh);
            double psi_b = qy_b(0) * (-0.5 * c.sh) + qy_b(3) * (0.5 * c.ch);
            double dpsi_b = 0.5 * a1.dot(qyd_b);
            psi_b += 0.25 * c.dpsi * a2.dot(qyd_b);
            double ddpsi_b = 0.5 * a1.dot(qydd_b);
            dpsi_b += 0.5 * c.dpsi * a2.dot(qydd_b);
            psi_b += 0.25 * c.ddpsi * a2.dot(qydd_b) -
                     0.125 * c.dpsi * c.dpsi * a1.dot(qydd_b);

            // The tilt part. lam0 = 1/d, lam1 = 1/d^3, lam3 = 1/d^5 with
            // d = sqrt(2 (1 + z_z)); collect the adjoints of the three and push
            // them into d once, at the end.
            double lam0_b = 0.0;
            double lam1_b = 0.0;
            double lam3_b = 0.0;

            const double b0 = qtdd_b(0), b1 = qtdd_b(1), b2 = qtdd_b(2);
            const double c0 = qtd_b(0), c1 = qtd_b(1), c2 = qtd_b(2);

            // t0dd = 0.5 lam0 zdd2 - 0.5 lam1 zd2^2
            lam0_b += 0.5 * zdd(2) * b0;
            zdd_b(2) += 0.5 * c.lam0 * b0;
            lam1_b += -0.5 * zd(2) * zd(2) * b0;
            zd_b(2) += -c.lam1 * zd(2) * b0;
            // t1dd = -lam0 zdd1 + z1 lam1 zdd2 + 2 lam1 zd1 zd2
            //        - 3 z1 lam3 zd2^2
            lam0_b += -zdd(1) * b1;
            zdd_b(1) += -c.lam0 * b1;
            z_b(1) += c.lam1 * zdd(2) * b1;
            lam1_b += z(1) * zdd(2) * b1;
            zdd_b(2) += z(1) * c.lam1 * b1;
            lam1_b += 2.0 * zd(1) * zd(2) * b1;
            zd_b(1) += 2.0 * c.lam1 * zd(2) * b1;
            zd_b(2) += 2.0 * c.lam1 * zd(1) * b1;
            z_b(1) += -3.0 * c.lam3 * zd(2) * zd(2) * b1;
            lam3_b += -3.0 * z(1) * zd(2) * zd(2) * b1;
            zd_b(2) += -6.0 * z(1) * c.lam3 * zd(2) * b1;
            // t2dd = lam0 zdd0 - z0 lam1 zdd2 - 2 lam1 zd0 zd2
            //        + 3 z0 lam3 zd2^2
            lam0_b += zdd(0) * b2;
            zdd_b(0) += c.lam0 * b2;
            z_b(0) += -c.lam1 * zdd(2) * b2;
            lam1_b += -z(0) * zdd(2) * b2;
            zdd_b(2) += -z(0) * c.lam1 * b2;
            lam1_b += -2.0 * zd(0) * zd(2) * b2;
            zd_b(0) += -2.0 * c.lam1 * zd(2) * b2;
            zd_b(2) += -2.0 * c.lam1 * zd(0) * b2;
            z_b(0) += 3.0 * c.lam3 * zd(2) * zd(2) * b2;
            lam3_b += 3.0 * z(0) * zd(2) * zd(2) * b2;
            zd_b(2) += 6.0 * z(0) * c.lam3 * zd(2) * b2;
            // t0d = 0.5 lam0 zd2
            lam0_b += 0.5 * zd(2) * c0;
            zd_b(2) += 0.5 * c.lam0 * c0;
            // t1d = -lam0 zd1 + z1 lam1 zd2
            lam0_b += -zd(1) * c1;
            zd_b(1) += -c.lam0 * c1;
            z_b(1) += c.lam1 * zd(2) * c1;
            lam1_b += z(1) * zd(2) * c1;
            zd_b(2) += z(1) * c.lam1 * c1;
            // t2d = lam0 zd0 - z0 lam1 zd2
            lam0_b += zd(0) * c2;
            zd_b(0) += c.lam0 * c2;
            z_b(0) += -c.lam1 * zd(2) * c2;
            lam1_b += -z(0) * zd(2) * c2;
            zd_b(2) += -z(0) * c.lam1 * c2;

            // lam0 = d^-1, lam1 = d^-3, lam3 = d^-5, d = sqrt(2 (1 + z2)).
            const double d = c.d;
            const double d_b = -c.lam0 * c.lam0 * lam0_b -
                               3.0 * c.lam1 * lam1_b / d -
                               5.0 * c.lam3 * lam3_b / d;
            z_b(2) += d_b / d;

            // qt = [0.5 d, -z1/d, z0/d, 0] carries its own z dependence.
            const Eigen::Vector4d &t_b = qt_b;
            const double d_b2 = 0.5 * t_b(0) + (z(1) / (d * d)) * t_b(1) -
                                (z(0) / (d * d)) * t_b(2);
            z_b(1) += -t_b(1) / d;
            z_b(0) += t_b(2) / d;
            z_b(2) += d_b2 / d;

            // zdd = v0 (Zbb - 2 u1 Zb - Tdd z + 2 u1 Td z)
            const double T = c.T;
            const double Td = c.Td;
            const double Tdd = c.Tdd;
            const double v0 = c.v0;
            const double u1 = c.u1;
            Eigen::Vector3d Zb_b = Eigen::Vector3d::Zero();
            Eigen::Vector3d Zbb_b = Eigen::Vector3d::Zero();
            double T_b = thrust_grad;
            double Td_b = 0.0;
            double Tdd_b = 0.0;
            double v0_b = 0.0;

            const Eigen::Vector3d t3 =
                c.Zbb - 2.0 * u1 * c.Zb - Tdd * z + 2.0 * u1 * Td * z;
            v0_b += t3.dot(zdd_b);
            const Eigen::Vector3d t3_b = v0 * zdd_b;
            Zbb_b += t3_b;
            const double u1_b = -2.0 * t3_b.dot(c.Zb) + 2.0 * Td * t3_b.dot(z);
            Zb_b += -2.0 * u1 * t3_b;
            Tdd_b += -t3_b.dot(z);
            z_b += -Tdd * t3_b;
            Td_b += 2.0 * u1 * t3_b.dot(z);
            z_b += 2.0 * u1 * Td * t3_b;

            // zd = v0 (Zb - Td z)
            Td_b += u1_b * v0;
            v0_b += u1_b * Td;
            const Eigen::Vector3d t4 = c.Zb - Td * z;
            v0_b += t4.dot(zd_b);
            const Eigen::Vector3d t4_b = v0 * zd_b;
            Zb_b += t4_b;
            Td_b += -t4_b.dot(z);
            z_b += -Td * t4_b;

            // v0 = 1 / T
            T_b += -v0_b / (T * T);

            // Tdd = (Zb.Zb)/T + z.Zbb - Td^2/T
            T_b += Tdd_b * (-c.Zb.dot(c.Zb) / (T * T));
            Zb_b += Tdd_b * (2.0 * c.Zb / T);
            z_b += Tdd_b * c.Zbb;
            Zbb_b += Tdd_b * z;
            T_b += Tdd_b * ((Td * Td) / (T * T));
            Td_b += Tdd_b * (-2.0 * Td / T);

            // Td = z . Zb
            z_b += Td_b * c.Zb;
            Zb_b += Td_b * z;

            Eigen::Vector3d jer_b = p.m * Zb_b;
            Eigen::Vector3d snap_b = p.m * Zbb_b;

            // T = m ||u||, z = u / ||u||
            const double r = c.r;
            const double r_b = p.m * T_b + (-z_b.dot(c.u) / (r * r));
            Eigen::Vector3d u_b = z_b / r + (r_b / r) * c.u;

            BackwardResult out;
            out.vel = Eigen::Vector3d::Zero();
            out.acc = u_b;
            out.jer = jer_b;
            out.snap = snap_b;
            out.psi = psi_b;
            out.dpsi = dpsi_b;
            out.ddpsi = ddpsi_b;
            return out;
        }

        FlappingParams params_;
        double thrust_eps = 1e-6;
    };
}  // namespace flapping

#endif
