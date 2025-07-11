import datetime
import logging
from enum import unique

import geopandas as gpd
import numpy as np
import xarray as xr
from lxml import etree
from sertit import geometry, path, rasters
from sertit.misc import ListEnum
from sertit.types import AnyPathStrType, AnyPathType

from eoreader import DATETIME_FMT, EOREADER_NAME, cache, utils
from eoreader.bands import (
    BLUE,
    GREEN,
    NARROW_NIR,
    NIR,
    RED,
    VRE_1,
    VRE_2,
    VRE_3,
    BandNames,
    SpectralBand,
)
from eoreader.bands.band_names import CA, DEEP_BLUE, WV, YELLOW, VenusMaskBandNames
from eoreader.exceptions import InvalidProductError
from eoreader.products import OpticalProduct
from eoreader.products.optical.optical_product import RawUnits
from eoreader.stac import CENTER_WV, FWHM, GSD, ID, NAME
from eoreader.utils import simplify

LOGGER = logging.getLogger(EOREADER_NAME)


@unique
class VenusProductType(ListEnum):
    """Venus products types (L2A)"""

    L2A = "VSC"
    """Level-2A: https://www.mdpi.com/2072-4292/14/14/3281"""


class VenusProduct(OpticalProduct):
    def __init__(
        self,
        product_path: AnyPathStrType,
        archive_path: AnyPathStrType = None,
        output_path: AnyPathStrType = None,
        remove_tmp: bool = False,
        **kwargs,
    ) -> None:
        # Initialization from the super class
        super().__init__(product_path, archive_path, output_path, remove_tmp, **kwargs)

    def _pre_init(self, **kwargs) -> None:
        """
        TODO : same as s2_theia_product
        """
        self._has_cloud_cover = True
        self.needs_extraction = False
        self._use_filename = True
        self._raw_units = RawUnits.REFL

        # Pre init done by the super class
        super()._pre_init(**kwargs)

    def _post_init(self, **kwargs) -> None:
        """
        TODO : same as s2_theia_product
        """
        self.tile_name = self._get_tile_name()

        # Post init done by the super class
        super()._post_init(**kwargs)

    def _get_name_constellation_specific(self) -> str:
        # Get MTD XML file
        root, _ = self.read_mtd()

        # Open identifier
        name = path.get_filename(root.findtext(".//IDENTIFIER"))
        if not name:
            raise InvalidProductError("IDENTIFIER not found in metadata!")

        return name

    @cache
    def _read_mtd(self) -> (etree._Element, dict):
        """
        Read metadata and outputs the metadata XML root and its namespaces as a dict

        .. code-block:: python

            >>> from eoreader.reader import Reader
            >>> path = r"VENUS-XS_20201029-105210-000_L2A_SUDOUE-1_C_V3-1.zip"
            >>> prod = Reader().open(path)
            >>> prod.read_mtd()
            (<Element Muscate_Metadata_Document at 0x252d2071e88>, {})

        Returns:
            (etree._Element, dict): Metadata XML root and its namespaces
        """

        # TODO : same as S2TheiaProduct
        mtd_from_path = "MTD_ALL.xml"
        mtd_archived = r"MTD_ALL\.xml"

        return self._read_mtd_xml(mtd_from_path, mtd_archived)

    def get_datetime(self, as_datetime: bool = False) -> str | datetime.datetime:
        """
        Get the product's acquisition datetime, with format :code:`YYYYMMDDTHHMMSS` <-> :code:`%Y%m%dT%H%M%S`

        .. code-block:: python

            >>> from eoreader.reader import Reader
            >>> path = r"VENUS-XS_20201029-105210-000_L2A_SUDOUE-1_C_V3-1"
            >>> prod = Reader().open(path)
            >>> prod.get_datetime(as_datetime=True)
            datetime.datetime(2020, 10, 29, 10, 52, 10, 000), fetched from metadata, so we have the ms
            >>> prod.get_datetime(as_datetime=False)
            '20201029-105210'

        Args:
            as_datetime (bool): Return the date as a datetime.datetime. If false, returns a string.

        Returns:
             Union[str, datetime.datetime]: Its acquisition datetime
        """

        # TODO : almost the same as S2TheiaProduct
        if self.datetime is None:
            # Get MTD XML file
            root, _ = self.read_mtd()

            # Open identifier
            acq_date = root.findtext(".//ACQUISITION_DATE")
            if not acq_date:
                raise InvalidProductError("ACQUISITION_DATE not found in metadata!")

            # Convert to datetime
            date = datetime.datetime.strptime(
                acq_date, "%Y-%m-%dT%H:%M:%S.%f"
            )  # no 'Z' at the end
        else:
            date = self.datetime

        if not as_datetime:
            date = date.strftime(DATETIME_FMT)
        return date

    def _get_tile_name(self) -> str:
        """
        TODO : same as s2_theia_product
        """
        # Get MTD XML file
        root, _ = self.read_mtd()

        # Open identifier
        tile = root.findtext(".//GEOGRAPHICAL_ZONE")
        if not tile:
            raise InvalidProductError("GEOGRAPHICAL_ZONE not found in metadata!")
        return tile

    def _set_instrument(self) -> None:
        """
        Set instrument

        VENµS : https://database.eohandbook.com/database/missionsummary.aspx?missionID=601&utm_source=eoportal&utm_content=venus
        """
        self.instrument = "VSC"

    def _set_product_type(self) -> None:
        """Set products type"""
        self.product_type = VenusProductType.L2A

    def _set_pixel_size(self) -> None:
        """
        Set product default pixel size (in meters)
        """
        self.pixel_size = 5.0

    def _map_bands(self) -> None:
        """
        Map bands
        """
        venus_bands = {
            DEEP_BLUE: SpectralBand(
                eoreader_name=DEEP_BLUE,
                **{NAME: "B1", ID: "1", GSD: 5, CENTER_WV: 420, FWHM: 40},
            ),
            CA: SpectralBand(
                eoreader_name=CA,
                **{NAME: "B2", ID: "2", GSD: 5, CENTER_WV: 443, FWHM: 40},
            ),
            BLUE: SpectralBand(
                eoreader_name=BLUE,
                **{NAME: "B3", ID: "3", GSD: 5, CENTER_WV: 490, FWHM: 40},
            ),
            GREEN: SpectralBand(
                eoreader_name=GREEN,
                **{NAME: "B4", ID: "4", GSD: 5, CENTER_WV: 555, FWHM: 40},
            ),
            YELLOW: SpectralBand(
                eoreader_name=YELLOW,
                **{NAME: "B5", ID: "5", GSD: 5, CENTER_WV: 620, FWHM: 40},
            ),
            RED: SpectralBand(
                eoreader_name=RED,
                **{NAME: "B7", ID: "7", GSD: 5, CENTER_WV: 667, FWHM: 30},
            ),
            VRE_1: SpectralBand(
                eoreader_name=VRE_1,
                **{NAME: "B8", ID: "8", GSD: 5, CENTER_WV: 702, FWHM: 24},
            ),
            VRE_2: SpectralBand(
                eoreader_name=VRE_2,
                **{NAME: "B9", ID: "9", GSD: 5, CENTER_WV: 742, FWHM: 16},
            ),
            VRE_3: SpectralBand(
                eoreader_name=VRE_3,
                **{NAME: "B10", ID: "10", GSD: 5, CENTER_WV: 782, FWHM: 16},
            ),
            NIR: SpectralBand(
                eoreader_name=NIR,
                **{NAME: "B11", ID: "11", GSD: 5, CENTER_WV: 865, FWHM: 40},
            ),
            NARROW_NIR: SpectralBand(
                eoreader_name=NARROW_NIR,
                **{NAME: "B11", ID: "11", GSD: 5, CENTER_WV: 865, FWHM: 40},
            ),
            WV: SpectralBand(
                eoreader_name=WV,
                **{NAME: "B12", ID: "12", GSD: 5, CENTER_WV: 910, FWHM: 20},
            ),
        }
        self.bands.map_bands(venus_bands)

    def _get_condensed_name(self) -> str:
        """
        Get products condensed name ({date}_VENUS_{tile]_{product_type}).

        Returns:
            str: Condensed name
        """
        return f"{self.get_datetime()}_VENUS_{self.tile_name}_{self.product_type.name}"

    def get_band_paths(
        self, band_list: list, pixel_size: float = None, **kwargs
    ) -> dict:
        """

        TODO : same as s2_theia
        TODO : not mandatory
        """
        band_paths = {}
        for band in band_list:  # Get clean band path
            clean_band = self.get_band_path(band, pixel_size=pixel_size, **kwargs)
            if clean_band.is_file():
                band_paths[band] = clean_band
            else:
                band_id = self.bands[band].id
                try:
                    if self.is_archived:
                        band_paths[band] = self._get_archived_rio_path(
                            rf".*FRE_B{band_id}\.tif"
                        )
                    else:
                        band_paths[band] = path.get_file_in_dir(
                            self.path, f"FRE_B{band_id}.tif"
                        )
                except (FileNotFoundError, IndexError) as ex:
                    raise InvalidProductError(
                        f"Non existing {band.name} ({band_id}) band for {self.path}"
                    ) from ex

        return band_paths

    def _load_bands(
        self,
        bands: list,
        pixel_size: float = None,
        size: list | tuple = None,
        **kwargs,
    ) -> dict:
        """
        TODO : same as s2_theia_product
        """
        # Return empty if no band are specified
        if not bands:
            return {}

        # Get band paths
        band_paths = self.get_band_paths(bands, pixel_size=pixel_size, **kwargs)

        # Open bands and get array (resampled if needed)
        band_arrays = self._open_bands(
            band_paths, pixel_size=pixel_size, size=size, **kwargs
        )

        return band_arrays

    def _read_band(
        self,
        band_path: AnyPathType,
        band: BandNames = None,
        pixel_size: tuple | list | float = None,
        size: list | tuple = None,
        **kwargs,
    ) -> xr.DataArray:
        """
        TODO : same as s2_theia_product
        """
        band_arr = utils.read(
            band_path,
            pixel_size=pixel_size,
            size=size,
            resampling=kwargs.pop("resampling", self.band_resampling),
            **kwargs,
        )

        # Convert type if needed
        if band_arr.dtype != np.float32:
            band_arr = band_arr.astype(np.float32)

        return band_arr

    def _to_reflectance(
        self,
        band_arr: xr.DataArray,
        band_path: AnyPathType,
        band: BandNames,
        **kwargs,
    ) -> xr.DataArray:
        """
        TODO : almost the same as s2_theia_product
        """
        # Compute the correct radiometry of the band for raw band
        if path.get_filename(band_path).startswith("VENUS"):
            band_arr /= 10000.0

        # Convert type if needed
        if band_arr.dtype != np.float32:
            band_arr = band_arr.astype(np.float32)

        return band_arr

    def get_quicklook_path(self) -> str:
        """
        TODO : same as s2_theia_product
        """
        quicklook_path = None
        try:
            if self.is_archived:
                quicklook_path = self.path / self._get_archived_path(
                    regex=r".*QKL_ALL\.jpg"
                )
            else:
                quicklook_path = next(self.path.glob("**/*QKL_ALL.jpg"))
            quicklook_path = str(quicklook_path)
        except (StopIteration, FileNotFoundError):
            LOGGER.warning(f"No quicklook found in {self.condensed_name}")

        return quicklook_path

    def _get_mask_path(self, mask_id: str) -> AnyPathType:
        """
        TODO : almost the same as s2_theia_product
        """
        mask_regex = f"*{mask_id}_XS.tif"  # XS
        try:
            if self.is_archived:
                mask_path = self._get_archived_rio_path(mask_regex.replace("*", ".*"))
            else:
                mask_path = path.get_file_in_dir(
                    self.path.joinpath("MASKS"), mask_regex, exact_name=True
                )
        except (FileNotFoundError, IndexError) as ex:
            raise InvalidProductError(
                f"Non existing mask {mask_regex} in {self.name}"
            ) from ex

        return mask_path

    @cache
    @simplify
    def footprint(self) -> gpd.GeoDataFrame:
        """
        TODO : almost the same as s2_theia_product
        """
        edg_path = self._get_mask_path(
            VenusMaskBandNames.EDG.name
        )  # there is no additional parameters
        mask = utils.read(edg_path, masked=False)

        # Vectorize the nodata band
        footprint = rasters.vectorize(mask, values=0, default_nodata=1)
        footprint = geometry.get_wider_exterior(footprint)
        footprint.geometry = footprint.geometry.convex_hull

        return footprint
