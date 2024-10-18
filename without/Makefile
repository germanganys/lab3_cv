CC= g++

all:
	$(CC) image_processing.cpp -o image_processing_o3 -I"/usr/include/opencv4" -L /usr/lib/aarch64-linux-gnu/ -lopencv_core -lopencv_highgui -lopencv_imgcodecs -O3
	$(CC) image_processing.cpp -o image_processing_o2 -I"/usr/include/opencv4" -L /usr/lib/aarch64-linux-gnu/ -lopencv_core -lopencv_highgui -lopencv_imgcodecs -O2
	$(CC) image_processing.cpp -o image_processing_o1 -I"/usr/include/opencv4" -L /usr/lib/aarch64-linux-gnu/ -lopencv_core -lopencv_highgui -lopencv_imgcodecs -O1
	$(CC) image_processing.cpp -o image_processing_o0 -I"/usr/include/opencv4" -L /usr/lib/aarch64-linux-gnu/ -lopencv_core -lopencv_highgui -lopencv_imgcodecs -O0
 
