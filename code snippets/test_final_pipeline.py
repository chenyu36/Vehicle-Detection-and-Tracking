# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

project_output = 'vehicle_detection_project_video.mp4'
input_clip = VideoFileClip("project_video.mp4")
output_clip = input_clip.fl_image(pipeline) #NOTE: this function expects color images!!
%time output_clip.write_videofile(project_output, audio=False)