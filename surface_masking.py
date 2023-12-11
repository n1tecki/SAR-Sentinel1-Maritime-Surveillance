# Required libraries
import geopandas as gpd
import numpy as np
import osmnx as ox
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape, LineString
from shapely.ops import polygonize
import tifffile

class SurfaceMaskProcessing:
    """
    A class for processing satellite images with associated geo-data.
    """

    def __init__(self, image_path: str, geojson_path: str, export_path: str, 
                 land_mask: bool = True, maritime_surface: bool = False):
        """
        Initializes the SatelliteImageProcessor with file paths.
        :param image_path: Path to the satellite image file.
        :param geojson_path: Path to the GeoJSON file.
        :param export_path: Path to export the processed image.
        """
        self.image_path = image_path
        self.geojson_path = geojson_path
        self.export_path = export_path
        self.src = None
        self.image_data = None
        self.image_profile = None
        self.polygon = None
        self.box = None
        self.land_mask = land_mask
        self.maritime_surface = maritime_surface

    def load_tiff(self):
        """
        Loads a TIFF image.
        """
        self.src = rasterio.open(self.image_path)
        self.image_data = self.src.read((1, 2, 3))
        self.image_profile = self.src.profile

    def get_coord(self):
        """
        Loads image coordinates from a GeoJSON file.
        """
        gdf = gpd.read_file(self.geojson_path)
        self.polygon, self.box = shape(gdf.loc[0, 'geometry']), gdf.total_bounds

    @staticmethod
    def closing_linestring(df: gpd.GeoDataFrame, polygon: shape, is_east_coast: bool, land_mask: bool, maritime_surface: bool) -> [LineString]:
        """
        Creates LineStrings to close open shapes.
        :param df: GeoDataFrame with geometry data.
        :param polygon: Polygon defining the area.
        :param is_east_coast: Boolean indicating if the area is on the east coast.
        :return: List of LineStrings.
        """
        max_north_list, max_south_list = [], []

        for linestring in df.geometry:
            x, y = linestring.coords.xy
            pairs = [list(z) for z in zip(x, y)]
            max_north_list.append(max(pairs, key=lambda v: v[1]))
            max_south_list.append(min(pairs, key=lambda v: v[1]))

        max_north_point = max(max_north_list, key=lambda v: v[1])
        max_south_point = min(max_south_list, key=lambda v: v[1])

        if is_east_coast:
          edge_x = polygon.bounds[0] if land_mask else polygon.bounds[2]
        else: 
          edge_x = polygon.bounds[2] if land_mask else polygon.bounds[0]

        return [
            LineString([(max_north_point[0], max_north_point[1]), (edge_x, max_north_point[1])]),
            LineString([(edge_x, max_north_point[1]), (edge_x, max_south_point[1])]),
            LineString([(edge_x, max_south_point[1]), (max_south_point[0], max_south_point[1])])
        ]

    @staticmethod
    def expand_gdf(dataframe: gpd.GeoDataFrame, additional_shapes: [LineString]) -> gpd.GeoDataFrame:
        """
        Adds LineStrings to an existing GeoDataFrame.
        :param dataframe: The original GeoDataFrame.
        :param additional_shapes: Shapes to be added.
        :return: Expanded GeoDataFrame.
        """
        template_row = {col: None for col in dataframe.columns}
        new_rows = [dict(template_row, geometry=line) for line in additional_shapes]

        new_rows_df = gpd.GeoDataFrame(new_rows)
        new_rows_df.crs = dataframe.crs
        return dataframe.append(new_rows_df, ignore_index=True)

    def masking_raster(self, polygon: shape) -> np.ndarray:
        """
        Applies masking to the raster image.
        :param polygon: Polygon to use for masking.
        :return: Masked image data.
        """
        polygon_masks = rasterize(
            [(geom, 1) for geom in polygon], 
            out_shape=self.src.shape, 
            transform=self.src.transform, 
            fill=0, 
            all_touched=True, 
            dtype='uint8'
        )
        polygon_masks = np.logical_not(polygon_masks)

        if len(self.image_data.shape) == 3:
            polygon_masks = np.stack([polygon_masks] * 3, axis=-1)
        return self.image_data * polygon_masks.transpose(2, 0, 1)

    def process_image(self):
        """
        Main method to process the satellite image.
        """
        self.load_tiff()
        self.get_coord()

        # Additional processing
        reference_longitude = 45.0792  # Center longitude coordinate of Saudi Arabia
        is_east_coast = self.polygon.centroid.x > reference_longitude
        coast_line = ox.features_from_polygon(self.polygon, tags={"natural": "coastline"})  # Extract coastline of port
        clipped_coast_line = coast_line.clip(self.polygon)  # Clip coast to image dimensions
        additional_shapes = self.closing_linestring(clipped_coast_line, self.polygon, is_east_coast, self.land_mask, self.maritime_surface)  # Create additional shapes to close area
        expanded_coast_line = self.expand_gdf(clipped_coast_line, additional_shapes)  # Expanding coastline with closing LineStrings

        mainland_polygon = gpd.GeoSeries(polygonize(expanded_coast_line.geometry))  # Making polygon from coastline
        mainland_polygon.set_crs(expanded_coast_line.crs, inplace=True)  # Setting the Coordinate Reference System

        masked_image = self.masking_raster(mainland_polygon)  # Masking the image
        tifffile.imwrite(self.export_path, masked_image)
