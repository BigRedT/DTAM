for i=1:13
    in_filename = ['/home/tanmay/Code/CameraCalibration/CalibImages/' num2str(i) '.jpg'];
    out_filename = ['/home/tanmay/Code/CameraCalibration/CalibImagesResized/' num2str(i) '.jpg'];
    img = imread(in_filename);
    resized_img = imresize(img, [1088 1920]);
    imwrite(resized_img, out_filename);
end
    
    

