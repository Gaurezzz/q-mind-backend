import mindspore_config 
import mindspore as ms
from mindspore.nn import Cell
from mindspore import ops

class BrusEngine(Cell):
    """
    BrusEngine: A physical modeling engine for Quantum Dot (QD) semiconductors.
    
    This class implements the Brus Equation to calculate the size-dependent energy 
    gap of nanoparticles, incorporating Varshni's Law for temperature correction 
    and a Gaussian distribution for spectral absorption profiles.
    
    Architecture: Modular / Decoupled
    Framework: MindSpore (Optimized for GPU/NPU acceleration)
    """

    def __init__(self, 
                 bandgap: float, 
                 alpha: float, 
                 beta: float, 
                 me_eff: float, 
                 mh_eff: float, 
                 eps_r: float, 
                 max_absorption_coefficient: float = 1e7):
        """
        Initializes the material-specific intrinsic properties.

        Args:
            bandgap (float): Bulk energy gap (E0) at 0 Kelvin [eV].
            alpha (float): Varshni thermal coefficient [eV/K].
            beta (float): Varshni constant related to Debye temperature [K].
            me_eff (float): Effective mass of the electron [relative to m0].
            mh_eff (float): Effective mass of the hole [relative to m0].
            eps_r (float): Relative dielectric constant of the material [dimensionless].
            max_absorption_coefficient (float): Peak absorption value [m^-1]. Default is 1e7.
        """
        super().__init__()
        
        # Material properties
        self.bandgap = bandgap
        self.alpha = alpha
        self.beta = beta
        self.me_eff = me_eff
        self.mh_eff = mh_eff
        self.eps_r = eps_r
        self.max_absorption_coefficient = max_absorption_coefficient
        
        # Pre-scaled physical constants (optimized for eV and nm units)
        # Avoids numerical underflow in float32 operations
        self.BRUS_CONST = 0.3760   # Confinement constant: h^2 / (8*m0) in [eV·nm^2]
        self.COUL_CONST = 2.5682   # Coulomb constant: 1.786*q / (4*pi*eps0) in [eV·nm]
        
        # Gaussian broadening parameter
        self.sigma = 10  # Standard deviation for absorption profile [nm]

    def construct(self, temperature: float, radius: float, wavelengths: ms.Tensor) -> ms.Tensor:
        """
        Computes the absorption coefficient spectrum based on the QD size and temperature.

        Args:
            temperature (float): Operating temperature [K].
            radius (float): Radius of the Quantum Dot [nm].
            wavelengths (ms.Tensor): A 1D tensor of wavelengths to evaluate [nm].

        Returns:
            ms.Tensor: Absorption coefficient profile [m^-1] for the given wavelength range.
        """
        
        # Varshni's Law: Temperature-dependent bandgap correction
        # Formula: Eg(T) = E0 - (alpha * T^2) / (T + beta)
        e_bulk = self.bandgap - (self.alpha * ops.pow(temperature, 2)) / (temperature + self.beta)

        # Brus Equation: Quantum confinement and Coulomb interaction (in eV and nm)
        confinement = (self.BRUS_CONST / ops.pow(radius, 2)) * (1/self.me_eff + 1/self.mh_eff)
        
        # Coulomb term: decreases energy gap due to electron-hole attraction
        coulomb = self.COUL_CONST / (self.eps_r * radius)
        
        # Total quantum dot energy gap [eV]
        e_qd = e_bulk + confinement - coulomb

        # Energy-to-Wavelength conversion: Peak absorption wavelength [nm]
        wavelength_peak = 1239.84 / e_qd

        # Gaussian absorption profile: Models size distribution effects
        absorption_coefficient = self.max_absorption_coefficient * ms.numpy.exp(
            -ops.pow((wavelengths - wavelength_peak), 2) / (2 * ops.pow(self.sigma, 2)) #type: ignore
        )

        return absorption_coefficient, e_qd #type: ignore