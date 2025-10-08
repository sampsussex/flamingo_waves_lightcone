import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from flamingo_photometry.galaxy_photometry_preparer import GalaxyPhotometryPreparer


@pytest.fixture
def mock_preparer():
    """Create a GalaxyPhotometryPreparer with mocked LightconeIO interfaces."""
    with patch("flamingo_photometry.galaxy_photometry_preparer.IndexedLightcone") as mock_lc, \
         patch("flamingo_photometry.galaxy_photometry_preparer.particle_halo_ids") as mock_phi:

        # Mock lightcone structure: pretend it has DM, Gas, and Stars particle groups
        lc_instance = MagicMock()
        lc_instance.basedir = "/mock/path"
        lc_instance.basename = "mock_lightcone"

        # Simulate property datasets
        lc_instance.__getitem__.side_effect = lambda ptype: MagicMock(
            properties={"Mass": np.arange(10),
                        "Velocity": np.random.rand(10, 3),
                        "Coordinates": np.random.rand(10, 3),
                        "Metallicity": np.random.rand(10)}
        )

        # Patch the assign_particle_halo_ids function
        mock_assignment = {
            "Gas": {"HaloID": np.array([1, 1, -1, 2, 2, -1, 3, 3, 3, -1])},
            "Stars": {"HaloID": np.array([1, -1, 1, 2, -1, 2, 3, -1, 3, 3])}
        }
        mock_phi.assign_particle_halo_ids.return_value = mock_assignment

        prep = GalaxyPhotometryPreparer(
            particle_lightcone_path="/mock/lightcone.hdf5",
            halo_lightcone_files=["/mock/halo_000.hdf5"],
            soap_catalog_files=["/mock/soap_000.hdf5"]
        )
        prep.lightcone = lc_instance
        yield prep


# ------------------------------------------------------------------------------
# TESTS
# ------------------------------------------------------------------------------

def test_build_galaxy_particle_map(mock_preparer):
    """Check that galaxy → particle mapping is correctly built."""
    prep = mock_preparer
    prep.build_galaxy_particle_map(["Gas", "Stars"])

    assert len(prep._galaxy_particle_map) == 3  # Expect 3 galaxies: 1, 2, 3
    assert "Gas" in prep._galaxy_particle_map[1]
    assert isinstance(prep._galaxy_particle_map[1]["Gas"], np.ndarray)


def test_get_galaxy_properties_success(mock_preparer):
    """Simulate galaxy property retrieval from mocked lightcone file."""
    prep = mock_preparer

    # Mock the open_file context and datasets
    fake_file = MagicMock()
    fake_file.__enter__.return_value = fake_file
    fake_file.__exit__.return_value = None
    fake_file["HaloID"].__getitem__.return_value = np.array([1, 2])
    fake_file["Mass"].__getitem__.return_value = np.array([1e10, 2e10])
    fake_file["Position"].__getitem__.return_value = np.array([[0, 0, 0], [1, 1, 1]])
    fake_file["Velocity"].__getitem__.return_value = np.array([[10, 0, 0], [20, 0, 0]])
    fake_file["Redshift"].__getitem__.return_value = np.array([0.1, 0.2])
    prep.lightcone.open_file.return_value = fake_file

    props = prep.get_galaxy_properties(1)
    assert props["GalaxyID"] == 1
    assert "Mass" in props and np.isclose(props["Mass"], 1e10).any()


def test_get_galaxy_properties_not_found(mock_preparer):
    """Check KeyError is raised if galaxy ID does not exist."""
    prep = mock_preparer

    fake_file = MagicMock()
    fake_file.__enter__.return_value = fake_file
    fake_file.__exit__.return_value = None
    fake_file["HaloID"].__getitem__.return_value = np.array([2, 3])
    prep.lightcone.open_file.return_value = fake_file

    with pytest.raises(KeyError):
        prep.get_galaxy_properties(999)


def test_get_galaxy_particle_properties(mock_preparer):
    """Verify per-galaxy particle property extraction for Gas + Stars."""
    prep = mock_preparer
    prep.build_galaxy_particle_map(["Gas", "Stars"])
    galaxy_id = 1

    result = prep.get_galaxy_particle_properties(
        galaxy_id,
        particle_props=["Mass", "Velocity", "Coordinates"]
    )

    assert "Gas" in result and "Stars" in result
    assert "Mass" in result["Gas"]
    assert isinstance(result["Gas"]["Mass"], np.ndarray)


def test_get_galaxy_particle_properties_no_map(mock_preparer):
    """Check RuntimeError is raised when map not built."""
    prep = mock_preparer
    prep._galaxy_particle_map = {}
    with pytest.raises(RuntimeError):
        prep.get_galaxy_particle_properties(1, ["Mass"])


def test_generate_photometry_placeholder(mock_preparer):
    """Ensure the placeholder raises NotImplementedError with correct args."""
    prep = mock_preparer

    gal_props = {"GalaxyID": 1, "Mass": [1e10]}
    gas_props = {"Mass": np.ones(5)}
    star_props = {"Mass": np.ones(3)}

    with pytest.raises(NotImplementedError):
        prep.generate_photometry(gal_props, gas_props, star_props)
