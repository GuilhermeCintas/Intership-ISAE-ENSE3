# import time

import numpy as np
import scipy


def transf_x(u):
    """
    Met sous forme qui autorise le produit matriciel

    (from scilab SUBFUNCTIONS/transf_x.sce)

    :param u: Vecteur [3]
    :return: Matrice [3, 3]
    """
    u_x = np.zeros((3, 3), dtype='float')
    u_x[0, 1] = - u[2]
    u_x[0, 2] = u[1]
    u_x[1, 0] = u[2]
    u_x[1, 2] = - u[0]
    u_x[2, 0] = - u[1]
    u_x[2, 1] = u[0]
    return u_x


def lr2sa(lr):
    """
    Commandes (Gauche, Droite) -> (Symmetrique, Antisymmetrique)

    (from scilab SIMFUNCTIONS/br2da.sce)

    :param lr: Vecteur [2]: Commandes (Gauche, Droite) dans [-1, +1]
    :return: Vecteur [2]: Commandes (Symmetrique dans [0, 1], Antisymmetrique dans [-1, +1])
    """
    sa = np.zeros_like(lr)
    sa[0] = 0.5 * (lr[0] + lr[1])
    sa[1] = 0.5 * (lr[0] - lr[1])
    return sa


def sa2lr(sa):
    """
    Commandes (Symmetrique, Antisymmetrique) -> (Gauche, Droite)

    :param sa: Vecteur [2]: Commandes (Symmetrique dans [0, 1], Antisymmetrique dans [-1, +1])
    :return: Vecteur [2]: Commandes (Gauche, Droite) dans [-1, +1]
    """
    lr = np.zeros_like(sa)
    lr[0] = sa[0] + sa[1]
    lr[1] = sa[0] - sa[1]
    return lr


# noinspection SpellCheckingInspection
class CoreOpale:
    """
    Contient les constantes et les calculs necessaires pour aboutir au calcul :
        dy = f(y)
    f etant self.dynamics_parachute()
    """

    def __init__(self):
        # # For profiling :
        # self.n_steps = 0
        # self.timing = 0.
        # self.print_at = 5.

        self.physics = {
            'gravity': 9.807,  # Gravite
            'air': 1.22401  # Masse volumique de l'air (alt = 0)
        }

        # SIMFUNCTIONS/set_parameters.sce - L69
        self.parafoil = {
            'thickness': 0.1105,  # Epaisseur
            'chord': 0.85,  # Corde
            'surface': 3.0,  # Aire
            'span': 3.4,  # Envergure
            'mass': 0.28,  # Masse du materiel
            'inertia': np.diag([0.2700, 0.0171, 0.2866]),  # Inertie du materiel
            'inertia_air': np.diag([0.4586,0.02911,0.4868]),  # Inertie de l'air inclus (alt = 0) TODO A CALCULER - m_air = 0.5*b*c*t*1.225 # I = I(mp+m-air)
            'from_link': np.array([0., 0., -1.9]),  # Vecteur entre C et P
            'mu': 0. * np.pi / 180.  # Angle de calage initial
        }
        self.parafoil['mass_air'] = (
                self.parafoil['span'] * self.parafoil['chord'] * self.parafoil['thickness'] * self.physics['air'] * 0.5
        )  # Masse de l'air inclus (alt = 0)
        self.parafoil['from_link_transf'] = transf_x(self.parafoil['from_link'])
        self.parafoil['cos_mu'] = np.cos(self.parafoil['mu'])
        self.parafoil['sin_mu'] = np.sin(self.parafoil['mu'])

        # SIMFUNCTIONS/set_parameters.sce - L85
        self.link = {
            'roll_stiffness': 10.,  # Raideur Roulis
            'roll_damping': 2.,  # Amortissement Roulis
            'pitch_stiffness': 0.,  # Raideur Tangage
            'pitch_damping': 0.,  # Amortissement Tangage
            'yaw_stiffness': 2.,  # Raideur Lacet
            'yaw_damping': 0.  # Amortissement Lacet
        }

        # SIMFUNCTIONS/set_parameters.sce - L93
        self.payload = {
            'mass': 10.,  # Masse
            'inertia': np.diag([0.15, 0.15, 0.15]),  # Inertie
            'drag': 1.,  # Trainee
            'from_link': np.array([0., 0., 0.3])  # Vecteur entre C et B
        }
        self.payload['from_link_transf'] = transf_x(self.payload['from_link'])

        # SIMFUNCTIONS/set_parameters.sce - L100
        self.apparent = {
            'd_roll_x': 0.,  # Distance entre P et le centre de roulis, sur x
            'd_roll_z': 1.,  # Distance entre P et le centre de roulis, sur z # TODO mis au hasard a revoir
            'd_pitch_x': 0.,  # Distance entre P et le centre de tangage, sur x
            'd_pitch_z': 0.,  # Distance entre P et le centre de tangage, sur z
            'mass': np.diag([0.0295, 0.1783, 1.5434]),  # Masse Apparente
            'inertia': np.diag([0.0696, 0.0440, 0.0314]),  # Inertie Apparente
            'distances': np.zeros((3, 3), dtype='float')  # Matrice des distances
        }
        self.apparent['distances'][0, 1] = - self.apparent['d_pitch_z']
        self.apparent['distances'][1, 0] = self.apparent['d_roll_z']
        self.apparent['distances'][1, 2] = - self.apparent['d_roll_x']
        self.apparent['distances'][2, 1] = self.apparent['d_pitch_x']

        # SIMFUNCTIONS/set_parameters.sce
        self.C = {
            # Coefficients longitudinaux
            'z': {
                '0': 0.33,
                'a': 3.5,
                'ds': 0.3197,
                'ds_nl': -0.1622,
                'da': 0.1  # Couplage
            },
            'x': {
                '0': 0.075,
                'Ki': 0.07,
                'ds': 0.0421,
                'ds_nl': 0.1854,
                'da': 0.02  # Couplage
            },
            'm': {
                '0': 0.,
                'a': 0.,
                'q': -0.200,
                'ds': 0.0116,
                'ds_nl': 0.0554,
            },

            # Coefficients lateraux
            'y': {
                'b': -0.05,
                'p': 0.,
                'r': 0.,
                'da': 0.02,
                'da_nl': 0.,
            },
            'l': {
                'b': 0.,
                'p': -0.1,
                'r': 0.04,
                'da': -0.129,
                'da_nl': 0.,
            },
            'n': {
                'b': 0.15,
                'p': -0.1,
                'r': -0.12,
                'da': 0.0484,
                'da_nl': 0.,
            }
        }

        self.vel = np.zeros(3, dtype='float')  # Vitesse (m/s)
        self.rot_p = np.zeros(3, dtype='float')  # Rotation voile (roulis/tangage/lacet, rad/s)
        self.rot_b = np.zeros(3, dtype='float')  # Rotation charge (roulis/tangage/lacet, rad/s)
        self.pos = np.zeros(3, dtype='float')  # Position (m)
        self.ang_p = np.zeros(3, dtype='float')  # Orientation voile (roulis/tangage/lacet, rad)
        self.ang_b = np.zeros(3, dtype='float')  # Orientation charge (roulis/tangage/lacet, rad)

        self.u_sym = self.u_asym = None  # Commandes Symmetrique [0, 1] et Asymmetrique [-1, +1]
        self.u_sym_nl = self.u_asym_nl = None  # Transformations non lineaires des commandes
        self.wind = np.zeros(3, dtype='float')  # Vent absolu (m/s)

        self.rho = None  # Masse volumique de l'air (kg/m3)

        self.cos_p = self.sin_p = self.tan_p = None  # Trigo de ang_p
        self.cos_b = self.sin_b = self.tan_b = None  # Trigo de ang_b

        self.PO = np.zeros((3, 3), dtype='float')  # Matrice de passage R_O -> R_P
        self.BO = np.zeros((3, 3), dtype='float')  # Matrice de passage R_O -> R_B
        self.BP = np.zeros((3, 3), dtype='float')  # Matrice de passage R_P -> R_B
        self.VP = np.zeros((3, 3), dtype='float')  # Matrice de passage R_P -> R_V
        self.VP[1, 1] = 1.

        self.R_P = np.zeros((3, 3), dtype='float')  # Matrice de rotation voile
        self.R_P[0, 0] = 1.
        self.R_B = np.zeros((3, 3), dtype='float')  # Matrice de rotation charge
        self.R_B[0, 0] = 1.

        self.rot_p_transf = np.zeros((3, 3), dtype='float')  # Vecteur rotation voile transforme en matrice
        self.rot_b_transf = np.zeros((3, 3), dtype='float')  # Vecteur rotation charge transforme en matrice

        self.mass_p = np.zeros((3, 3), dtype='float')  # Masse voile
        self.iner_p = np.zeros((3, 3), dtype='float')  # Intertie voile
        self.grav_p = np.zeros(3, dtype='float')  # Gravitation voile
        self.grav_b = np.zeros(3, dtype='float')  # Gravitation charge

        self.vel_p = np.zeros(3, dtype='float')  # Vitesse dans le repere voile
        self.mass_app = np.zeros((3, 3), dtype='float')  # Masse apparente
        self.iner_app = np.zeros((3, 3), dtype='float')  # Intetie apparente
        self.dist_app = np.zeros((3, 3), dtype='float')  # Distances apparentes

        self.ang_bp = np.zeros(3, dtype='float')  # Orientation voile / charge (roulis/tangage/lacet, rad)
        self.rot_bp = np.zeros(3, dtype='float')  # Rotation voile / charge (roulis/tangage/lacet, rad/s)
        self.link_torq = np.zeros(3, dtype='float')  # Moments a la liaison rotule

        self.wind_norm_v = None  # Norme du vent dans le repere R_V
        self.alpha = self.beta = None  # Angles aerodynamiques
        self.coef_linear = np.zeros(3, dtype='float')  # Coefficients de forces frottements
        self.coef_angular = np.zeros(3, dtype='float')  # Coefficients de moments frottements

        self.R_AE = np.zeros((3, 3), dtype='float')  # # Matrice de rotation aerodynamique
        self.drag_p = np.zeros(3, dtype='float')  # Force de frottements voile
        self.torq_p = np.zeros(3, dtype='float')  # Moment de frottements voile
        self.drag_b = np.zeros(3, dtype='float')  # Force de frottements charge
        self.torq_b = np.zeros(3, dtype='float')  # Moment de frottements charge

        self.A = np.zeros((12, 12), dtype='float')  # AX = B cote gauche
        self.B = np.zeros(12, dtype='float')  # AX = B cote droit
        self.dy = np.zeros(18, dtype='float')  # Return : variation de l'etat

    def atmosphere(self):
        """
        Masse volumique de l'air en fonction de l'altitude

        (atmosphereISA.sce)

        Requires : [pos]

        Updates : [rho]
        """
        # Atmosphere standard :
        # Psol = 101325 Pa
        # Tsol = 288.15 K (soit 15 C)

        # P = 101325. * np.exp(-0.0001185 * z)
        # T = 288.15 - 0.0065 * altitude
        # rho = P * 0.0034837 / T

        # -> Polynomial regression
        altitude = -self.pos[2]
        self.rho = (4.6174e-09 * altitude - 1.15009e-04) * altitude + self.physics['air']

    def nonlinear_command(self):
        """
        Transformation non-lineaire sur les commandes symmetrique et asymmetrique

        Requires : [u_sym, u_asym]

        Updates : [u_sym_nl, u_asym_nl]
        """
        self.u_sym_nl = max(0., self.u_sym - 0.5) ** 2
        self.u_asym_nl = np.sign(self.u_asym) * max(0., abs(self.u_asym) - 0.5) ** 2

    def trigonometry(self):
        """
        Fonctions trigonometriques des angles

        Requires : [ang_p, ang_b]

        Updates : [cos_p, sin_p, tan_p, cos_b, sin_b, tan_b]
        """
        self.cos_p = np.cos(self.ang_p)
        self.sin_p = np.sin(self.ang_p)
        self.tan_p = np.tan(self.ang_p)
        self.cos_b = np.cos(self.ang_b)
        self.sin_b = np.sin(self.ang_b)
        self.tan_b = np.tan(self.ang_b)

    def transformation_matrices(self):
        """
        Matrices de passage entre les referentiels Rp, Rb, Ro.

        (SUBFUNCTIONS/transformation_matrices.sce)

        Requires : [cos_p, sin_p, cos_b, sin_b]

        Updates : [PO, BO, BP, VP]
        """
        # Ro -> Rp
        self.PO[0, 0] = self.cos_p[1] * self.cos_p[2]
        self.PO[0, 1] = self.cos_p[1] * self.sin_p[2]
        self.PO[0, 2] = - self.sin_p[1]
        self.PO[1, 0] = self.sin_p[0] * self.sin_p[1] * self.cos_p[2] - self.cos_p[0] * self.sin_p[2]
        self.PO[1, 1] = self.sin_p[0] * self.sin_p[1] * self.sin_p[2] + self.cos_p[0] * self.cos_p[2]
        self.PO[1, 2] = self.sin_p[0] * self.cos_p[1]
        self.PO[2, 0] = self.cos_p[0] * self.sin_p[1] * self.cos_p[2] + self.sin_p[0] * self.sin_p[2]
        self.PO[2, 1] = self.cos_p[0] * self.sin_p[1] * self.sin_p[2] - self.sin_p[0] * self.cos_p[2]
        self.PO[2, 2] = self.cos_p[0] * self.cos_p[1]

        # Ro -> Rb
        self.BO[0, 0] = self.cos_b[1] * self.cos_b[2]
        self.BO[0, 1] = self.cos_b[1] * self.sin_b[2]
        self.BO[0, 2] = - self.sin_b[1]
        self.BO[1, 0] = self.sin_b[0] * self.sin_b[1] * self.cos_b[2] - self.cos_b[0] * self.sin_b[2]
        self.BO[1, 1] = self.sin_b[0] * self.sin_b[1] * self.sin_b[2] + self.cos_b[0] * self.cos_b[2]
        self.BO[1, 2] = self.sin_b[0] * self.cos_b[1]
        self.BO[2, 0] = self.cos_b[0] * self.sin_b[1] * self.cos_b[2] + self.sin_b[0] * self.sin_b[2]
        self.BO[2, 1] = self.cos_b[0] * self.sin_b[1] * self.sin_b[2] - self.sin_b[0] * self.cos_b[2]
        self.BO[2, 2] = self.cos_b[0] * self.cos_b[1]

        # Rp -> Rb
        self.BP = np.dot(self.BO,self.PO.T)

        # Rp -> Rv
        self.VP[0, 0] = self.parafoil['cos_mu']
        self.VP[0, 2] = - self.parafoil['sin_mu']
        self.VP[2, 0] = self.parafoil['sin_mu']
        self.VP[2, 2] = self.parafoil['cos_mu']

    def rotation_matrices(self):
        """
        Matrices de rotation voile et charge

        (SUBFUNCTIONS/mkTbTp.sce)

        Requires : [cos_p, sin_p, tan_p, cos_b, sin_b, tan_b, rot_p, rot_b]

        Updates : [R_P, R_B, rot_p_transf, rot_b_transf]
        """
        assert abs(self.cos_p[1]) > 1e-10 and abs(self.cos_b[1]) > 1e-10, "Gimbal lock, pitch = 90"

        self.R_P[0, 1] = self.tan_p[1] * self.sin_p[0]
        self.R_P[0, 2] = self.tan_p[1] * self.cos_p[0]
        self.R_P[1, 1] = self.cos_p[0]
        self.R_P[1, 2] = - self.sin_p[0]
        self.R_P[2, 1] = self.sin_p[0] / self.cos_p[1]
        self.R_P[2, 2] = self.cos_p[0] / self.cos_p[1]

        self.R_B[0, 1] = self.tan_b[1] * self.sin_b[0]
        self.R_B[0, 2] = self.tan_b[1] * self.cos_b[0]
        self.R_B[1, 1] = self.cos_b[0]
        self.R_B[1, 2] = - self.sin_b[0]
        self.R_B[2, 1] = self.sin_b[0] / self.cos_b[1]
        self.R_B[2, 2] = self.cos_b[0] / self.cos_b[1]

        self.rot_p_transf[0, 1] = - self.rot_p[2]
        self.rot_p_transf[0, 2] = self.rot_p[1]
        self.rot_p_transf[1, 0] = self.rot_p[2]
        self.rot_p_transf[1, 2] = - self.rot_p[0]
        self.rot_p_transf[2, 0] = - self.rot_p[1]
        self.rot_p_transf[2, 1] = self.rot_p[0]

        self.rot_b_transf[0, 1] = - self.rot_b[2]
        self.rot_b_transf[0, 2] = self.rot_b[1]
        self.rot_b_transf[1, 0] = self.rot_b[2]
        self.rot_b_transf[1, 2] = - self.rot_b[0]
        self.rot_b_transf[2, 0] = - self.rot_b[1]
        self.rot_b_transf[2, 1] = self.rot_b[0]

    def mass_and_inertia(self):
        """
        Masse et Inertie du parachute
        Forces de gravitation sur le parachute et la charge

        (SIMFUNCTIONS/mass_inertia.sce)

        Requires : [rho, VP, PO, BO]

        Updates : [mass_p, iner_p, grav_p, grav_b]
        """
        mass_with_air = self.parafoil['mass'] + self.rho * self.parafoil['mass_air'] / self.physics['air']
        inertia_with_air = self.parafoil['inertia'] + self.rho * self.parafoil['inertia_air'] / self.physics['air']

        self.mass_p[0, 0] = mass_with_air
        self.mass_p[1, 1] = mass_with_air
        self.mass_p[2, 2] = mass_with_air

        self.iner_p = np.dot(self.VP.T, np.dot(inertia_with_air,self.VP))

        self.grav_p = self.PO[:, 2] * self.parafoil['mass'] * self.physics['gravity']
        self.grav_b = self.BO[:, 2] * self.payload['mass'] * self.physics['gravity']

    def apparent_mass_and_inertia(self):
        """
        Matrice de masse apparente, Matrice d'inertie apparente
        Matrice des distances aux centres de roulis, tangage, lacet

        (SIMFUNCTIONS/apparent_mass_inertia.sce)

        Requires : [vel, PO, VP, rho]

        Updates : [vel_p, dist_app, mass_app, iner_app]
        """
        # changement de repere de Rv a Rp
        self.vel_p = np.dot(self.PO , self.vel)
        self.dist_app = np.dot(self.VP.T , np.dot(self.apparent['distances'], self.VP))
        self.mass_app = self.rho * np.dot(self.VP.T, np.dot(self.apparent['mass'], self.VP))
        self.iner_app = (self.rho * np.dot(self.VP.T, np.dot(self.apparent['inertia'], np.dot(self.VP
                         - self.dist_app.T, np.dot(self.mass_app, self.dist_app)))))

    def link_torque(self):
        """
        Moments au point de liaison rotule

        Requires : [BP, rot_p, rot_b]

        Updates : [ang_bp, rot_bp, link_torq]
        """
        self.ang_bp[0] = np.arctan2(self.BP[1, 2], self.BP[2, 2])
        self.ang_bp[1] = -np.arcsin(self.BP[0, 2])
        self.ang_bp[2] = np.arctan2(self.BP[0, 1], self.BP[0, 0])

        self.rot_bp = self.rot_b - np.dot(self.BP , self.rot_p)

        self.link_torq[0] = (- self.link['roll_stiffness'] * self.ang_bp[0]
                             - self.link['roll_damping'] * self.rot_bp[0])
        self.link_torq[1] = (- self.link['pitch_damping'] * self.rot_bp[1]
                             - self.link['pitch_stiffness'] * self.ang_bp[1])
        self.link_torq[2] = (- self.link['yaw_stiffness'] * self.ang_bp[2]
                             - self.link['yaw_damping'] * self.rot_bp[2])

    def aero_coeff(self):
        """
        Coefficients aerodynamiques : lineaires et angulaires

        (SIMFUNCTIONS/model_aeroNL.sce)

        Requires : [VP, rot_p, wind_norm_v, alpha, beta, u_sym, u_sym_nl, u_asym, u_asym_nl]

        Updates : [coef_linear, coef_angular]
        """
        rot_v = np.dot(self.VP , self.rot_p)
        p = rot_v[0] * self.parafoil['span'] / (2 * self.wind_norm_v)
        q = rot_v[1] * self.parafoil['chord'] / (2 * self.wind_norm_v)
        r = rot_v[2] * self.parafoil['span'] / (2 * self.wind_norm_v)

        # Equations des coefficients longitudinaux
        c_z = (
                self.C['z']['0'] +
                self.C['z']['a'] * self.alpha +
                self.C['z']['ds'] * self.u_sym +
                self.C['z']['ds_nl'] * self.u_sym_nl +
                self.C['z']['da'] * abs(self.u_asym)
        )

        c_x = (
                self.C['x']['0'] +
                self.C['x']['Ki'] * c_z ** 2 +
                self.C['x']['ds'] * self.u_sym +
                self.C['x']['ds_nl'] * self.u_sym_nl +
                self.C['x']['da'] * abs(self.u_asym)
        )

        c_m = (
                self.C['m']['0'] +
                self.C['m']['a'] * self.alpha +
                self.C['m']['ds'] * self.u_sym +
                self.C['m']['ds_nl'] * self.u_sym_nl +
                self.C['m']['q'] * q
        )
        # Equations des coefficients lateraux
        c_y = (
                self.C['y']['b'] * self.beta +
                self.C['y']['p'] * p +
                self.C['y']['r'] * r +
                self.C['y']['da'] * self.u_asym +
                self.C['y']['da_nl'] * self.u_asym_nl
        )
        c_l = (
                self.C['l']['b'] * self.beta +
                self.C['l']['p'] * p +
                self.C['l']['r'] * r +
                self.C['l']['da'] * self.u_asym +
                self.C['l']['da_nl'] * self.u_asym_nl
        )
        c_n = (
                self.C['n']['b'] * self.beta +
                self.C['n']['p'] * p +
                self.C['n']['r'] * r +
                self.C['n']['da'] * self.u_asym +
                self.C['n']['da_nl'] * self.u_asym_nl
        )
        self.coef_linear[0] = - c_x
        self.coef_linear[1] = c_y
        self.coef_linear[2] = - c_z
        self.coef_angular[0] = c_l * self.parafoil['span']
        self.coef_angular[1] = c_m * self.parafoil['chord']
        self.coef_angular[2] = c_n * self.parafoil['span']

    def aerodynamics(self):
        """
        Efforts aerodynamiques : forces et moments

        (SIMFUNCTIONS/aerodynamics.sce)

        Requires : [VP, PO, BO, vel, wind, rot_p, rot_b, u_sym, u_sym_nl, u_asym, u_asym_nl, rho]

        Updates : [drag_p, torq_p, drag_b, torq_b, wind_norm_v, alpha, beta, R_AE, coef_linear, coef_angular]
        """
        wind_vel_v = np.dot(self.VP , ( np.dot(self.PO , (self.vel - self.wind)) - np.dot(self.parafoil['from_link_transf'], self.rot_p)))
        wind_vel_b = np.dot(self.BO , (self.vel - self.wind)) - np.dot(self.payload['from_link_transf'] , self.rot_b)

        self.wind_norm_v = np.linalg.norm(wind_vel_v)

        self.alpha = np.arctan2(wind_vel_v[2], wind_vel_v[0])
        self.beta = np.arctan2(wind_vel_v[1], wind_vel_v[0])

        # Trigo
        cos_alpha = np.cos(self.alpha)
        sin_alpha = np.sin(self.alpha)
        cos_beta = np.cos(self.beta)
        sin_beta = np.sin(self.beta)

        # determination des coefficients aerodynamiques
        self.aero_coeff()

        # calculs des forces et moments aerodynamiques
        self.R_AE[0, 0] = cos_alpha * cos_beta
        self.R_AE[0, 1] = - cos_alpha * sin_beta
        self.R_AE[0, 2] = - sin_alpha
        self.R_AE[1, 0] = sin_beta
        self.R_AE[1, 1] = cos_beta
        self.R_AE[2, 0] = sin_alpha * cos_beta
        self.R_AE[2, 1] = - sin_alpha * sin_beta
        self.R_AE[2, 2] = cos_alpha

        w = 0.5 * self.rho * self.parafoil['surface'] * self.wind_norm_v ** 2
        self.drag_p = w * np.dot(self.VP.T, np.dot(self.R_AE , self.coef_linear))
        self.torq_p = w * np.dot(self.VP.T, self.coef_angular)
        self.drag_b = - 0.5 * self.rho * self.payload['drag'] * np.linalg.norm(wind_vel_b) * wind_vel_b

    def dynamics_parachute(self, y, u, wind):
        """
        Variation dy = f(y)

        (SIMFUNCTIONS/dynamics_parachute.sce)

        :param y: Vecteur [18]: Etat du systeme [pos, ang_p, ang_b, vel, rot_p, rot_b]
        :param u: Vecteur [2]: Commandes (Gauche, Droite) dans [-1, +1]
        :param wind: Vecteur [3]: [Vx, Vy, Vz]
        :return: Vecteur [18]: Variation dy = f(y)
        """

        # # For profiling :
        # start_timer = time.time()

        self.pos, self.ang_p, self.ang_b, self.vel, self.rot_p, self.rot_b = np.split(y, 6)
        self.u_sym, self.u_asym = lr2sa(u)
        self.wind = wind

        self.atmosphere()
        self.nonlinear_command()
        self.trigonometry()

        self.transformation_matrices()  # 6 % of iteration time
        self.rotation_matrices()

        self.mass_and_inertia()  # 5 % of iteration time
        self.apparent_mass_and_inertia()  # 6 % of iteration time

        self.link_torque()
        self.aerodynamics()  # 21 % of iteration time

        # Solve A @ X = B

        # Fill A
        # 10 % of iteration time
        # A1
        self.A[0:3, 0:3] = self.payload['mass'] * self.BO
        self.A[0:3, 6:9] = - self.payload['mass'] * self.payload['from_link_transf']
        self.A[0:3, 9:12] = - self.BO
        # A2
        self.A[3:6, 6:9] = self.payload['inertia']
        self.A[3:6, 9:12] = np.dot(self.payload['from_link_transf'] , self.BO)
        # A3
        self.A[6:9, 0:3] = np.dot((self.mass_p + self.mass_app) , self.PO)
        self.A[6:9, 3:6] = - ( np.dot(self.mass_p , self.parafoil['from_link_transf'])
                              + np.dot(self.mass_app , (self.parafoil['from_link_transf'] + self.dist_app)))
        self.A[6:9, 9:12] = self.PO
        # A4
        self.A[9:12, 0:3] = - np.dot(self.dist_app.T , np.dot(self.mass_app, self.PO))
        self.A[9:12, 3:6] = (self.iner_p + self.iner_app
                             + np.dot(self.dist_app.T, np.dot(self.mass_app , self.parafoil['from_link_transf'])))
        self.A[9:12, 9:12] = - np.dot(self.parafoil['from_link_transf'] , self.PO)

        # Fill B
        # 14 % of iteration time
        # B1
        self.B[0:3] = (self.drag_b + self.grav_b
                       + self.payload['mass'] * np.dot(self.rot_b_transf , np.dot(self.payload['from_link_transf'], self.rot_b)))
        # B2
        self.B[3:6] = (self.torq_b + self.link_torq
                       - np.dot(self.rot_b_transf , np.dot(self.payload['inertia'], self.rot_b)))
        # B3
        self.B[6:9] = (
                self.drag_p + self.grav_p
                - np.dot(self.rot_p_transf, np.dot(self.A[6:9, 3:6], self.rot_p))
                + np.dot((np.dot(self.mass_app , self.rot_p_transf) - np.dot(self.rot_p_transf, self.mass_app)) , self.vel_p)
        )
        # B4
        self.B[9:12] = (
                self.torq_p
                - np.dot(self.BP.T , self.link_torq)
                - np.dot(self.rot_p_transf , np.dot(self.A[9:12, 0:3], self.vel))
                - np.dot(self.rot_p_transf , np.dot(self.A[9:12, 3:6], self.rot_p))
                - np.dot(self.dist_app.T, np.dot(self.mass_app, np.dot(self.rot_p_transf, self.vel_p)))
                - np.dot(transf_x(np.dot(self.mass_app, np.dot(self.dist_app, self.rot_p))),(
                        self.vel_p - np.dot(self.parafoil['from_link_transf'], self.rot_p)
                ))
        )

        x = scipy.linalg.solve(self.A, self.B)  # 19 % of iteration time

        self.dy[0:3] = self.vel
        self.dy[3:6] = np.dot(self.R_P, self.rot_p)
        self.dy[6:9] = np.dot(self.R_B, self.rot_b)
        self.dy[9:18] = x[0:9]

        # # For profiling :
        # self.n_steps += 1
        # self.timing += time.time() - start_timer
        # if self.timing > self.print_at:
        #     self.print_at += 5.
        #     print(1e6 * self.timing / self.n_steps)

        return self.dy
