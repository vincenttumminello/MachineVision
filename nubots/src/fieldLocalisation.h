/**
 * @file fieldLocalisation.h
 * @brief Offline RoboCup field localisation pipeline from recorded robot data.
 */
#ifndef FIELDLOCALISATION_H
#define FIELDLOCALISATION_H

#include <filesystem>

/**
 * @brief Run field localisation on a recorded NUbots data set.
 *
 * Expects dataDir to contain recorded_data.json and Left_timecode.txt
 * (and optionally Left.mp4 for visualisation).
 *
 * @param dataDir Directory containing the recorded data
 * @param interactive Interactivity level (0: none, 1: pause on last frame, 2: pause on all frames)
 * @param outputDirectory Directory for exported results (empty: no export)
 */
void runFieldLocalisation(const std::filesystem::path & dataDir, int interactive, const std::filesystem::path & outputDirectory);

#endif
