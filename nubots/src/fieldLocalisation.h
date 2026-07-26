/**
 * @file fieldLocalisation.h
 * @brief Offline RoboCup field localisation pipeline from recorded robot data.
 */
#ifndef FIELDLOCALISATION_H
#define FIELDLOCALISATION_H

#include <filesystem>
#include <string>

/**
 * @brief Run field localisation on a recorded NUbots data set.
 *
 * Expects dataDir to contain recorded_data.json and Left_timecode.txt
 * (and optionally Left.mp4 for visualisation).
 *
 * @param dataDir Directory containing the recorded data
 * @param interactive Interactivity level (0: none, 1: pause on last frame, 2: pause on all frames)
 * @param outputDirectory Directory for exported results (empty: no export)
 * @param lensName Camera calibration to replay with (see CameraLens.h); empty
 *                 selects it from the recorded frame size
 * @param fieldName Field the recording was made on (see FieldMap.h); empty
 *                  follows the camera calibration
 */
void runFieldLocalisation(const std::filesystem::path & dataDir, int interactive, const std::filesystem::path & outputDirectory,
                          const std::string & lensName = {}, const std::string & fieldName = {});

#endif
