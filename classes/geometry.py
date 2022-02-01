from polylidar import MatrixDouble, Polylidar3D
from shapely.geometry import Polygon, Point

class Geometry:
    
    # https://jeremybyu.github.io/polylidar/python_api/polylidar.Polylidar3D.html#polylidar.Polylidar3D.extract_planes_and_polygons
    def extract_polygons(self, points, polylidar_kwargs={}):
        return Polylidar3D(**polylidar_kwargs).extract_planes_and_polygons(MatrixDouble(points))[2]
    
    # https://jeremybyu.github.io/polylidar/python_api/polylidar.Polylidar3D.html#polylidar.Polylidar3D.extract_planes_and_polygons
    def extract_polygon_indexes(self, points, polylidar_kwargs={}):
        list = []
        for p in self.extract_polygons(points, polylidar_kwargs=polylidar_kwargs):
            list.append(p.shell)
        return list
    
    # https://shapely.readthedocs.io/en/stable/manual.html#polygons
    def create_polygon(self, points, indexes):
        list = []
        for i in indexes:
            list.append(points[i])
        return Polygon(list)
    
    # https://shapely.readthedocs.io/en/stable/manual.html#polygons
    def get_points_in_polygon(self, points, polygon):
        list = []
        for i in points:
            if polygon.contains(Point(i)):
                list.append(tuple(i))
        return list
    
    # https://shapely.readthedocs.io/en/stable/manual.html#polygons
    def get_indexes_of_points_in_polygon(self, points, indexes, polygon):
        list = []
        for i, point in enumerate(points):
            if polygon.contains(Point(point)):
                list.append(indexes[i])
        return list