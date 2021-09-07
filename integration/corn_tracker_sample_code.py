from tracking_opticalflow_func import corn_tracking_opticalflow
from tracking_sort_func import corn_tracking_sort


"""
corn_dict will have {frame_number1:{id1:[(x1,y1),(x2,y2)], id2:[(x1,y1),(x2,y2)], ..}, frame_number2:..}
"""

corn_dict = corn_tracking_opticalflow('/home/jungseok/Downloads/offline_frames/side_color')
corn_dict = corn_tracking_sort('/home/jungseok/Downloads/offline_frames/side_color', output_file=None)
