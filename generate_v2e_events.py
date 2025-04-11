import math
import random
import os
import shutil
import cv2


def generateVideoFromSingleImage(input_img_path,input_img_name,destination_video_path,output_dim = (376,290)):
    
    import numpy as np
    import cv2
    
    src_fname = input_img_name
    src_path = input_img_path
    dest_path = destination_video_path
    src_fname_name = src_fname.rsplit('.', 1)[0]
    print("src name is ",src_fname)
    src_fname_ext = src_fname.rsplit('.',1)[1]
    rotXdeg = -0.5
    rotYdeg = 0.5
    rotZdeg = 0
    f = 300
    dist = 300

    saccade_1 = [(-0.5,0,0,0),(-0.4,-0.1,-2,2),(-0.3,-0.2,-4,4),(-0.2,-0.3,-6,6),(-0.1,-0.4,-8,8),(0,-0.5,-10,10)]
    saccade_2 = [(0.1,-0.4,-8,8),(0.2,-0.3,-6,6),(0.3,-0.2,-4,4),(0.4,-0.1,-2,2),(0.5,0,0,0)]
    saccade_3 = [(0.4,0.1,2,2),(0.3,0.2,4,4),(0.2,0.3,6,6),(0.1,0.4,8,8),(0,0.5,10,10)]
    saccade_4 = [(-0.1,0.4,8,8),(-0.2,0.3,6,6),(-0.3,0.2,4,4),(-0.4,0.1,2,2),(-0.5,0,0,0)]
    
    saccade_points = saccade_1 + saccade_2 + saccade_3 + saccade_4
    #Read input image, and create output image
    src_full_name = src_path + '/' + src_fname
    src = cv2.imread(src_full_name)
    if src is None:
        return -1

    dst = np.ndarray(shape=src.shape,dtype=src.dtype)
    
    
    #Create user interface with trackbars that will allow to modify the parameters of the transformation
    wndname1 = "Source:"
    wndname2 = "WarpPerspective: "
    
    h,w = src.shape[:2]
    print("height and width are = ",h, " ",w);
    if (h > w):
        src = cv2.rotate(src, cv2.ROTATE_90_CLOCKWISE);
    
    #src = cv2.resize(src,output_dim, interpolation = cv2.INTER_CUBIC);
    h,w = src.shape[:2]
    print("height and width are 2 = ",h, " ",w);
    #aspect_ratio = 1.49;
    #print("aspect ratio = ",aspect_ratio);
    #while(True):
    output_dim = (376,290)
   
    #h = 376/aspect_ratio
    #h = math.ceil(h)
    #w = 376
    #output_dim = (w,h)

    src = cv2.resize(src,output_dim, interpolation = cv2.INTER_CUBIC);
    h , w = src.shape[:2]
    print("resized height and width are = ",h, " ",w);
    numOfSaccadePoints = len(saccade_points)
    count = 0
    retVal = False
    #cv2.imshow(wndname1, src)
    for i in range(2):
        if count > 25:
                retVal = True

        for j in range(numOfSaccadePoints):
            if count > 25:
                retVal = True

            idx = j
            if (i == 1):
                idx = numOfSaccadePoints -1 - j # idx = 10 - 2 - j
            rotXdeg = 6*saccade_points[idx][0]
            rotYdeg = 6*saccade_points[idx][1]
            tx = saccade_points[idx][2]/2
            ty = saccade_points[idx][3]/2
            print(rotXdeg,' ',rotYdeg)
            rotX = (rotXdeg)*np.pi/180
            rotY = (rotYdeg)*np.pi/180
            rotZ = (rotZdeg)*np.pi/180

            #Projection 2D -> 3D matrix
            A1= np.matrix([[1, 0, -w/2],
                            [0, 1, -h/2],
                            [0, 0, 0   ],
                            [0, 0, 1   ]])

            # Rotation matrices around the X,Y,Z axis
            RX = np.matrix([[1,           0,            0, 0],
                                [0,np.cos(rotX),-np.sin(rotX), 0],
                                [0,np.sin(rotX),np.cos(rotX) , 0],
                                [0,           0,            0, 1]])

            RY = np.matrix([[ np.cos(rotY), 0, np.sin(rotY), 0],
                                [            0, 1,            0, 0],
                                [ -np.sin(rotY), 0, np.cos(rotY), 0],
                                [            0, 0,            0, 1]])

            RZ = np.matrix([[ np.cos(rotZ), -np.sin(rotZ), 0, 0],
                                [ np.sin(rotZ), np.cos(rotZ), 0, 0],
                                [            0,            0, 1, 0],
                                [            0,            0, 0, 1]])

            #Composed rotation matrix with (RX,RY,RZ)
            R = RX * RY * RZ
            
            #Translation matrix on the Z axis change dist will change the height
            T = np.matrix([[1,0,0,tx],
                            [0,1,0,ty],
                            [0,0,1,dist],
                            [0,0,0,1]])

            #Camera Intrisecs matrix 3D -> 2D
            A2= np.matrix([[f, 0, w/2,0],
                            [0, f, h/2,0],
                            [0, 0,   1,0]])

            # Final and overall transformation matrix
            H = A2 * (T * (R * A1))

            # Apply matrix transformation
            warped_image = cv2.warpPerspective(src, H,output_dim,cv2.INTER_CUBIC)  
            w = output_dim[0] 
            h = output_dim[1]
            #print(len(warped_image[0]))
            warped_image = warped_image[15:(h-15),15:(w-15)]
            print('dest shape',warped_image.shape[:2])
            #Show the image
            #cv2.imshow(wndname2, dst)
            dest_full_name = dest_path + '/' + src_fname_name + '_' + str(idx) + '_' + str(i) + '.' + src_fname_ext
            cv2.imwrite(dest_full_name,warped_image)
            count += 1
            cv2.waitKey(10)
            
    return retVal

def generateEventSequenceForGivenCondition(input_video_path,output_folder,condition,event_h5_fname):
    thre_low, thre_high = 0.05, 0.5
    sigma_low, sigma_high = 0, 0
    cutoff_hz_low, cutoff_hz_high = 0, 0
    leak_rate_hz_low, leak_rate_hz_high = 0, 0
    shot_noise_rate_hz_low, shot_noise_rate_hz_high = 0, 0
    if condition == "ideal":
        thre_low, thre_high = 0.05, 0.5
        sigma_low, sigma_high = 0, 0
        cutoff_hz_low, cutoff_hz_high = 0, 0
        leak_rate_hz_low, leak_rate_hz_high = 0, 0
        shot_noise_rate_hz_low, shot_noise_rate_hz_high = 0, 0
    elif condition == "bright":
        thre_low, thre_high = 0.05, 0.5
        sigma_low, sigma_high = 0.03, 0.05
        cutoff_hz_low, cutoff_hz_high = 200, 200
        leak_rate_hz_low, leak_rate_hz_high = 0.1, 0.5
        shot_noise_rate_hz_low, shot_noise_rate_hz_high = 0, 0
    elif condition == "dark":
        thre_low, thre_high = 0.05, 0.5
        sigma_low, sigma_high = 0.03, 0.05
        cutoff_hz_low, cutoff_hz_high = 10, 100
        leak_rate_hz_low, leak_rate_hz_high = 0, 0
        shot_noise_rate_hz_low, shot_noise_rate_hz_high = 1, 10
    
    thres = 0.1
    sigma = random.uniform(
        min(thres*0.15, sigma_low), min(thres*0.25, sigma_high)) \
        if condition != "ideal" else 0
    leak_rate_hz = random.uniform(leak_rate_hz_low, leak_rate_hz_high)
    shot_noise_rate_hz = random.uniform(
    shot_noise_rate_hz_low, shot_noise_rate_hz_high)
    if condition == "dark":
        # cutoff hz follows shot noise config
        cutoff_hz = shot_noise_rate_hz*10
    else:
        cutoff_hz = random.uniform(cutoff_hz_low, cutoff_hz_high)
    
    v2e_command = [
        "v2e ",
        "-i", input_video_path,
        "-o", output_folder,
        "--overwrite",
        "--unique_output_folder", "false",
        "--no_preview",
        #"--skip_video_output",
        "--disable_slomo",
        "--pos_thres", "{}".format(thres),
        "--neg_thres", "{}".format(thres),
        "--sigma_thres", "{}".format(sigma),
        "--cutoff_hz", "{}".format(cutoff_hz),
        "--leak_rate_hz", "{}".format(leak_rate_hz),
        "--shot_noise_rate_hz", "{}".format(shot_noise_rate_hz),
        "--input_frame_rate", "42",
        "--input_slowmotion_factor", "10",
        "--dvs_h5", event_h5_fname,
        "--dvs_text", "None",
        "--dvs_exposure", "duration", "0.01",
        "--auto_timestamp_resolution", "false",
        "--dvs346"]
    
    final_v2e_command = " ".join(v2e_command)
    print(final_v2e_command)
    os.system(final_v2e_command)
    #!$final_v2e_command


def generateEventSequenceForAllConditions(input_video_path,output_folder,only_events_folder_path_for_category,event_h5_fname):
    condition_list = ["ideal","bright","dark"]
    retVal = False
    for condition in condition_list:
        print("generating events for ",condition)
        output_folder_for_condition = output_folder + "/_" + condition
        if not os.path.exists(output_folder_for_condition):
            os.makedirs(output_folder_for_condition)
        generateEventSequenceForGivenCondition(input_video_path,output_folder_for_condition,condition,event_h5_fname)
        only_events_folder_path_for_category_condition = only_events_folder_path_for_category + "/_" + condition

        ##Collecting Event sequences seperately

        if not os.path.exists(only_events_folder_path_for_category_condition):
            os.makedirs(only_events_folder_path_for_category_condition)
        
        src_fname_to_copy = output_folder_for_condition + "/" + event_h5_fname + ".h5"
        if os.path.exists(src_fname_to_copy):
            dest_fname_to_copy = only_events_folder_path_for_category_condition + "/" + event_h5_fname + ".h5"
            print("copying event file from ",src_fname_to_copy, " to ",dest_fname_to_copy)
            shutil.copy(src_fname_to_copy,dest_fname_to_copy)
            shutil.rmtree(output_folder_for_condition)
            retVal = True
        else:
            retVal = False

    return retVal


def generateEventsForAGivenImageCategory(src_path,destination_video_path,destination_events_path,category_name,only_event_folder_path):
    
    output_dim_for_video = (376,290)
    image_full_names_list = []
    img_id = 0
    for img_name in os.listdir(src_path):
        #if img_id == 5:
        #    break
        output_video_folder_path = destination_video_path + "/" + category_name + "/" + category_name + "_" + str(img_id)
        if not os.path.exists(output_video_folder_path):
            os.makedirs(output_video_folder_path)
        
        retVal = generateVideoFromSingleImage(src_path,img_name,output_video_folder_path,output_dim = output_dim_for_video)
        if not retVal:
            return True

        output_event_sequence_folder_path = destination_events_path + "/" + category_name + "/" + category_name + "_" + str(img_id)
        if not os.path.exists(output_event_sequence_folder_path):
            os.makedirs(output_event_sequence_folder_path)

        only_event_foder_path_for_category = only_event_folder_path + "/" + category_name
        if not os.path.exists(only_event_foder_path_for_category):
            os.makedirs(only_event_foder_path_for_category)
        
        src_fname_name = img_name.rsplit('.', 1)[0]
        event_h5_fname = src_fname_name + "_" + str(img_id)
        retVal2 = generateEventSequenceForAllConditions(output_video_folder_path,output_event_sequence_folder_path,only_event_foder_path_for_category,event_h5_fname)
        img_id += 1
        if retVal2:
            shutil.rmtree(output_video_folder_path)

def main():


    #Corresponding to each static image we generate three event-camera simulated videos of 50ms length so that one video is noise_free/ideal, second one is 'dark' mode events and other 
    #one is "bright" mode events. More description about each of these modes are available in v2e paper "https://sites.google.com/view/video2events/home?pli=1"
    
    
    damage_category_list = ["healthy","spalling","crack"]

    for damage_category in damage_category_list:
    	#'src_path' is the folder where you have images of different damage categories from which a event-camera simulated video should be generated.
        src_path = "/media/dtu-neurorobotics-desk2/data_2/CCSC_Database_Crack_Spalling_compare/healthy/"
        #'destination_video_path' is the folder that contains a set of transformed images
        destination_video_path = "/media/dtu-neurorobotics-desk2/data_2/CCSC_Database_Crack_Spalling_compare/Collected_events/Image_to_Video_data_collect/"
        #'destination_events_path' contains the files including event videos generated with v2e
        destination_events_path = "/media/dtu-neurorobotics-desk2/data_2/CCSC_Database_Crack_Spalling_compare/Collected_events/Event_Data_Collector/Event_full_data/"
    	#'only_events_folder_path' contains the event video files generated by v2e (we copied only these files from destination_events_path' . All the event video files are 
    	#in ".h5" format 
        only_events_folder_path = "/media/dtu-neurorobotics-desk2/data_2/CCSC_Database_Crack_Spalling_compare/Collected_events/Event_Data_Collector/Only_events/"

        if (damage_category == "crack"):
            src_path = src_path + "/crack"
            category_name = "crack"
            
        elif (damage_category == "spalling"):
            src_path = src_path + "/spalling"
            category_name = "spalling"

        elif (damage_category == "healthy"):
            src_path = src_path + "/healthy"
            category_name = "healthy"
        else:
            print("continueing since this image doesn't belong to any category ....")
            continue

        generateEventsForAGivenImageCategory(src_path,destination_video_path,destination_events_path,category_name,only_events_folder_path)

if __name__ == '__main__':
    main()
