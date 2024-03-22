import os, shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt


def compare_frames(previous_frame, current_frame):
    diff = current_frame - previous_frame
    return np.mean(diff, dtype=np.int64)
    

def main():
    videos = [f for f in os.listdir(".") if f.endswith(".MP4")]

    for v in videos:
        # get file path for desired video and where to save frames locally
        path_to_save = os.path.join(os.path.join(".", v.replace(".MP4", "")))
        if os.path.exists(path_to_save):
            shutil.rmtree(path_to_save)
        os.makedirs(path_to_save)

        cap = cv2.VideoCapture(v)
        if (cap.isOpened() == False):
            print('Cap is not open')

        frame_diff = []
        previous_frame = None
        current_frame = None 
        frame_counter = 0
        # cap opened successfully
        while(cap.isOpened()):
            # capture each frame
            ret, frame = cap.read()
            if(ret == True):
                previous_frame = current_frame
                current_frame = frame
                frame_counter += 1
                #if current_frame % 10 == 0:
                #    # Save frame as a jpg file
                #    file_name = str(current_frame).zfill(6) + '.png'
                #    print(f'Saving {os.path.join(path_to_save, file_name)}')
                #    cv2.imwrite(os.path.join(path_to_save, file_name), frame)

                # keep track of how many images you end up with
                #previous_frame = current_frame
                #current_frame += 1    
            else:
                break

            print(f"Video {v} Frame counter {frame_counter}")
            if previous_frame is not None:
                frame_diff.append(compare_frames(previous_frame, current_frame))
            
            if frame_counter > 5000:
                break

        print(len(frame_diff))

        # release capture 
        cap.release()
        print('done')
        # Plot frame_diff
        plt.plot(frame_diff)
        plt.xlabel('Frame Number')
        plt.ylabel('Frame Difference')
        plt.title('Frame Difference over Time')
        plot_name = v.replace(".MP4", "")
        plt.savefig(f"{plot_name}_plot.png")
        plt.close()

if __name__ == '__main__':
    main()