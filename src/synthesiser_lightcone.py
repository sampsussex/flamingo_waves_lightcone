import numpy as np
import logging
from typing import Dict, List, Any
from collections import defaultdict

from lightcone_io.particle_reader import IndexedLightcone
from lightcone_io import particle_halo_ids


# ------------------------------------------------------------------------------
# Configure a module-level logger
# ------------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)


class GalaxyPhotometryPreparer:
    """
    Prepares per-galaxy gas and stellar particle data from a Flamingo light cone,
    suitable for passing to a photometry generation routine.

    Steps:
    1. Load Flamingo lightcone via LightconeIO.
    2. Map particles (gas, star) → galaxy IDs using LightconeIO halo assignment.
    3. Extract galaxy-level and particle-level data for one galaxy at a time.
    4. Provide a placeholder for the photometry generation step.
    """

    def __init__(self,
                 particle_lightcone_path: str,
                 halo_lightcone_files: List[str],
                 soap_catalog_files: List[str],
                 so_name: str = "SO/200_crit",
                 overlap_method: str = "fractional_radius"):
        """
        Initialize the preparer with paths to Flamingo lightcone data.

        Parameters
        ----------
        particle_lightcone_path : str
            Path to the Flamingo particle lightcone file (HDF5).
        halo_lightcone_files : list[str]
            List of halo lightcone HDF5 files.
        soap_catalog_files : list[str]
            List of SOAP catalog HDF5 files.
        so_name : str
            Spherical overdensity definition to use for halo radius.
        overlap_method : str
            Method for assigning particles to halos when overlaps occur.
        """
        self.lightcone = IndexedLightcone(particle_lightcone_path)
        self.halo_lightcone_files = halo_lightcone_files
        self.soap_catalog_files = soap_catalog_files
        self.so_name = so_name
        self.overlap_method = overlap_method

        # Mapping: GalaxyID -> {particle_type: indices}
        self._galaxy_particle_map: Dict[int, Dict[str, np.ndarray]] = {}

    # --------------------------------------------------------------------------
    # TASK 1: Map particles to galaxy IDs
    # --------------------------------------------------------------------------
    def build_galaxy_particle_map(self, particle_types: List[str] = ["Gas", "Stars"]) -> None:
        """
        Build a mapping of galaxy (halo) IDs → particle indices for each particle type.

        Parameters
        ----------
        particle_types : list[str]
            List of particle types to include in the mapping.
        """
        logger.info("Assigning particles to halos using LightconeIO...")

        # The LightconeIO function will return per-particle halo IDs.
        # This part assumes the assignment can be obtained in-memory (no file writing).
        assignment = particle_halo_ids.assign_particle_halo_ids(
            lightcone_dir=self.lightcone.basedir,
            lightcone_basename=self.lightcone.basename,
            halo_lightcone_files=self.halo_lightcone_files,
            soap_files=self.soap_catalog_files,
            so_name=self.so_name,
            overlap_method=self.overlap_method,
        )

        # Suppose the assignment returns a dict: {ptype: {"HaloID": np.array([...])}}
        galaxy_map = defaultdict(lambda: defaultdict(list))

        for ptype in particle_types:
            halo_ids = assignment[ptype]["HaloID"]
            for idx, gid in enumerate(halo_ids):
                if gid < 0:
                    continue
                galaxy_map[int(gid)][ptype].append(idx)

        # Convert lists to numpy arrays
        self._galaxy_particle_map = {
            gid: {ptype: np.array(indices, dtype=int)
                  for ptype, indices in ptypedict.items()}
            for gid, ptypedict in galaxy_map.items()
        }

        logger.info(f"Built galaxy-particle mapping for {len(self._galaxy_particle_map)} galaxies.")

    # --------------------------------------------------------------------------
    # TASK 2: Collect galaxy-level properties
    # --------------------------------------------------------------------------
    def get_galaxy_properties(self, galaxy_id: int) -> Dict[str, Any]:
        """
        Retrieve the halo/galaxy properties for a given galaxy ID.

        Parameters
        ----------
        galaxy_id : int
            The galaxy (halo) ID.

        Returns
        -------
        dict[str, Any]
            Dictionary of galaxy-level properties.
        """
        for halo_file in self.halo_lightcone_files:
            with self.lightcone.open_file(halo_file) as f:
                ids = f["HaloID"][()]
                mask = ids == galaxy_id
                if np.any(mask):
                    props = {
                        "GalaxyID": galaxy_id,
                        "Mass": f["Mass"][()][mask],
                        "Position": f["Position"][()][mask],
                        "Velocity": f["Velocity"][()][mask],
                        "Redshift": f["Redshift"][()][mask],
                    }
                    logger.debug(f"Loaded properties for galaxy {galaxy_id}")
                    return props
        raise KeyError(f"Galaxy ID {galaxy_id} not found in halo catalogs.")

    # --------------------------------------------------------------------------
    # TASK 3: Collect particle-level properties (Gas + Stars)
    # --------------------------------------------------------------------------
    def get_galaxy_particle_properties(self,
                                       galaxy_id: int,
                                       particle_props: List[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Retrieve the requested particle properties for Gas and Stars belonging to one galaxy.

        Parameters
        ----------
        galaxy_id : int
            The galaxy (halo) ID.
        particle_props : list[str]
            Names of particle properties to extract (e.g. ["Mass", "Coordinates", "Velocity", "Metallicity"]).

        Returns
        -------
        dict[str, dict[str, np.ndarray]]
            Nested dictionary of the form:
            {
                "Gas": {"Mass": [...], "Velocity": [...], ...},
                "Stars": {"Mass": [...], "Velocity": [...], ...}
            }
        """
        if not self._galaxy_particle_map:
            raise RuntimeError("Galaxy-particle map not built. Run build_galaxy_particle_map() first.")

        galaxy_entry = self._galaxy_particle_map.get(galaxy_id)
        if galaxy_entry is None:
            raise KeyError(f"No particles found for Galaxy ID {galaxy_id}")

        result = {}
        for ptype, indices in galaxy_entry.items():
            result[ptype] = {}
            for prop in particle_props:
                dataset = self.lightcone[ptype].properties[prop]
                result[ptype][prop] = dataset[indices]
            logger.debug(f"Loaded {ptype} properties for galaxy {galaxy_id}")
        return result

    # --------------------------------------------------------------------------
    # TASK 4: Placeholder — Photometry Generation
    # --------------------------------------------------------------------------
    def generate_photometry(self,
                            galaxy_properties: Dict[str, Any],
                            gas_particle_properties: Dict[str, np.ndarray],
                            star_particle_properties: Dict[str, np.ndarray]) -> None:
        """
        Placeholder for generating photometry from galaxy and particle data.

        Parameters
        ----------
        galaxy_properties : dict[str, Any]
            Dictionary of global galaxy properties.
        gas_particle_properties : dict[str, np.ndarray]
            Gas particle properties (mass, metallicity, etc.).
        star_particle_properties : dict[str, np.ndarray]
            Stellar particle properties (mass, age, metallicity, etc.).

        Raises
        ------
        NotImplementedError
            This function must be implemented by the user for photometric modeling.
        """
        raise NotImplementedError("Photometry generation is not implemented yet.")


# ------------------------------------------------------------------------------
# Example usage (testing structure only)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    particle_file = "./lightcone_particles/lightcone_0000.0.hdf5"
    halo_files = ["lightcone_halos_0000.hdf5"]
    soap_files = ["soap_props_0000.hdf5"]

    prep = GalaxyPhotometryPreparer(particle_file, halo_files, soap_files)
    prep.build_galaxy_particle_map(["Gas", "Stars"])

    # Pick a galaxy
    galaxy_id = next(iter(prep._galaxy_particle_map.keys()))

    # Get galaxy + particle properties
    gal_props = prep.get_galaxy_properties(galaxy_id)
    part_props = prep.get_galaxy_particle_properties(
        galaxy_id,
        particle_props=["Mass", "Velocity", "Coordinates", "Metallicity"]
    )

    # Pass to placeholder photometry function
    prep.generate_photometry(gal_props,
                             gas_particle_properties=part_props["Gas"],
                             star_particle_properties=part_props["Stars"])
