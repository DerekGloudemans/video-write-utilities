
import requests
import os 


temp_name = "/home/derek/Desktop/mega_output_test.mp4"
f_name = "/home/derek/Desktop/mega_output_test2.mp4"
os.system("/usr/bin/ffmpeg -i {} -vcodec libx264 {}".format(temp_name,f_name))


url = 'http://viz-dev.isis.vanderbilt.edu:5991/upload?type=extra'
files = {'upload_file': open(f_name,'rb')}
ret = requests.post(url, files=files)
print(f_name)
print(ret)
if ret.status_code == 200:
    print('Uploaded!')
