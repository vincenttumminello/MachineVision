#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>
#include <filesystem>
#include "DJIVideoCaption.h"

std::vector<DJIVideoCaption> getVideoCaptions(const std::filesystem::path & captionPath)
{
    assert(std::filesystem::exists(captionPath));

    std::vector<DJIVideoCaption> caps;

    // Open the SRT file
    FILE* file = fopen(captionPath.string().c_str(), "r");
    if (!file) {
        return caps;
    }
    
    char line[2048];
    
    while (!feof(file)) {
        // Read subtitle number
        int subtitleNum;
        if (fscanf(file, "%d\n", &subtitleNum) != 1) {
            break;
        }
        
        // Read and parse timestamp line to get time
        int startH, startM, startS, startMs;
        int endH, endM, endS, endMs;
        if (fscanf(file, "%d:%d:%d,%d --> %d:%d:%d,%d\n",
                   &startH, &startM, &startS, &startMs,
                   &endH, &endM, &endS, &endMs) != 8) {
            break;
        }
        
        // Calculate time in seconds from start timestamp
        double time = startH * 3600.0 + startM * 60.0 + startS + startMs / 1000.0;
        
        // Read the first line with FrameCnt and date/time
        int frameCnt, diffTime;
        int year, month, day, hour, min, sec, msec, usec;
        if (fscanf(file, "<font size=\"36\">FrameCnt : %d, DiffTime : %dms\n",
                   &frameCnt, &diffTime) != 2) {
            break;
        }
        
        if (fscanf(file, "%d-%d-%d %d:%d:%d,%d,%d\n",
                   &year, &month, &day, &hour, &min, &sec, &msec, &usec) != 8) {
            break;
        }
        
        // Read the second line with camera parameters and GPS data
        int iso, fnum, ev, ct, focal_len;
        float shutter;
        char color_md[64];
        double latitude, longitude, altitude;
        
        if (fgets(line, sizeof(line), file) == nullptr) {
            break;
        }
        
        int parsed = sscanf(line,
            "[iso : %d] [shutter : 1/%f] [fnum : %d] [ev : %d] [ct : %d] [color_md : %[^]]] [focal_len : %d] "
            "[latitude : %lf] [longtitude : %lf] [altitude: %lf]",
            &iso, &shutter, &fnum, &ev, &ct, color_md, &focal_len,
            &latitude, &longitude, &altitude);
        
        if (parsed >= 10) {  // All essential fields parsed
            DJIVideoCaption caption;
            caption.frameNum = frameCnt;
            caption.time = time;
            caption.iso = iso;
            caption.shutterHz = shutter;  // Already in Hz (e.g., 120 for 1/120)
            caption.fnum = fnum / 100.0;  // Convert from 280 to 2.8
            caption.latitude = latitude;
            caption.longitude = longitude;
            caption.altitude = altitude;
            
            caps.push_back(caption);
        }
        
        // Read blank line between entries (and consume any remaining line content)
        while (fgets(line, sizeof(line), file) != nullptr) {
            // Check if we hit a blank line or next entry
            if (line[0] == '\n' || line[0] == '\r' || 
                (line[0] >= '0' && line[0] <= '9')) {
                // If it's a number, we need to "put it back" by seeking back
                if (line[0] >= '0' && line[0] <= '9') {
                    fseek(file, -(long)strlen(line), SEEK_CUR);
                }
                break;
            }
        }
    }
    
    fclose(file);
    return caps;
}
