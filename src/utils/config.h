//Spicoli Piersilvio

#ifndef CONFIG_H
#define CONFIG_H

#include <string>

inline const std::string TENSORFLOW_MODEL_PATH    = "../models/CNN_model.pb";
inline const std::string WINDOW_NAME              = "Face detection and emotion recognition";
inline const std::string HAAR_CASCADE_FRONTALFACE_PATH   = "../models/haarcascade_frontalface_alt2.xml";
inline const std::string HAAR_CASCADE_PROFILEFACE_PATH    = "../models/haarcascade_profileface.xml";
inline const std::string OUTPUT_DIR               = "../output";

// Settings for resizing windows for images that are too large
const int WIN_W = 1280, WIN_H = 720;    

#endif //CONFIG_H
