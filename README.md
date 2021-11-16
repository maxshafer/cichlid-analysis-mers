# cichlid-analysis
cichlid behaviour analysis code 

General pipeline
Get units:
Run measure_units.py (assumes you're measuring the. width of the divider which is divider_base_mm = 15) on each camera folder's background.
Run measure_fish.py (allows. you to select a frame to measure the standard length of the fish)

Run run_fish_als.py (creates initial plots to allow you to see how well the fish is tracked)

Quality control:
Run tracker_checker.py (a GUI for looking at the tracking of videos you think might not have trackeed well - generally if thresholded plot has a lot of black)

Now you can:
1. divide_tracking.py (allows you to retrack movies with different background images created from that video with the defined time bins)
2. split_tracking.py (allows you to exclude a period of the tracking, this is often necessary if there was a bump. to the camera, water level got too low and we can see when it was added etc).
3. rretracking.py (can change the ROI, generate new backgrounds etc).

After you have gone through the videos, you now only need the X and Y files. We move. thesee to an  analysis folder which have these files for each fish of that species.

Now you can run
run_combine_fish.als (creates combined data and bins data for that species, also generates the Z file)

You can move the Z file to a folder which has these files for all species. This will allow you to run:
run_down_sampled.py (for plots and analysis across fish species)
