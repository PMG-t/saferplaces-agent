import numpy as np
import leafmap.maplibregl as leafmap

from ..common import utils, s3_utils

class LeafmapInterface():
    
    def __init__(self):
        self.m = leafmap.Map()
        self.registred_layers = []  # DOC: Only src by noww ..
        
        
    def add_layer(self, src, layer_type, **kwargs):
        
        if src in self.registred_layers:
            return            
        
        if layer_type == 'vector':
            self.add_vector_layer(src, **kwargs)
        
        elif layer_type == 'raster':
            self.add_raster_layer(src, **kwargs)
        
        else:
            raise ValueError(f'Layer type {layer_type} is not supported. Valid layer types are ["vector", "raster"]')
        
        self.registred_layers.append(src)
        return True
        
        
    def add_vector_layer(self, src, **kwargs):
        """Add a vector layer to the map."""
        
        src = utils.s3uri_to_https(src)
        name = kwargs.pop('title', utils.juststem(src))
        
        self.m.add_vector(
            data = src,
            name = name
        )
        
    def add_raster_layer(self, src, **kwargs):
        """Add a raster layer to the map."""
        
        src_cog = utils.tif_to_cog(src)
        
        src_cog = utils.s3uri_to_https(src_cog)
        name = kwargs.pop('title', utils.juststem(src_cog))
        colormap = kwargs.pop('colormap_name', 'blues')
        nodata = kwargs.pop('nodata', -9999)
        
        self.m.add_cog_layer(
            url = src_cog,
            name = name,
            colormap_name = colormap,
            nodata = nodata,
        )
        
        