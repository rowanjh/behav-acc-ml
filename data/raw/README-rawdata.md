# Table of Contents

1.  [Overview](#overview)

2.  [Zip 1: raw-data-db.zip](#raw-data-db.zip)

    1.  [acc table](#ruff-acc.db:%20acc%20table)
    2.  [recordings table](#ruff-acc.db:%20recordings%20table)

3.  [Zip 2: raw-data-other.zip](#raw-data-other.zip)

    1.  [logger deployment notes csv](#logger_deployment_notes.csv)
    2.  [ruff behaviour tidy csv](#ruff_behaviour_tidy_2022-11-16.csv)
    3.  [calibration recordings csv](#calibration_recordings_6o_apr2022.csv)
    4.  [6O calibration files](#6o_calibration_files)

# Overview {#overview}

This readme describes the dataset for the following paper:

Aulsebrook, Jacques-Hamilton, & Kempenaers (2023) Quantifying mating behaviour using accelerometry and machine learning: challenges and opportunities. Animal Behaviour, 2023

Dataset available at <https://edmond.mpdl.mpg.de/privateurl.xhtml?token=3c5da533-d851-4988-994c-f984c5f1b1a4>

Scripts to accompany the analysis are available at <https://github.com/rowanjh/behav-acc-ml>

The dataset includes two zip files, `raw-data-db.zip`, and `raw-data-other.zip`, described below

------------------------------------------------------------------------

# raw-data-db.zip {#raw-data-db.zip}

------------------------------------------------------------------------

This zip contains an sqlite3 database 'ruff-acc.db' approx 80GB in size. This database has 30 raw accelerometery recordings from captive Ruffs. Each recording contains approximately 5-8 days of raw accelerometer data sampled at 50Hz.

Raw accelerometer data from loggers were converted to csv using software from the accelerometer manufacturers. Raw csv files were subsequently imported into an sqlite3 database. Original csv files are not released in the public dataset.

### ruff-acc.db: acc table

This table contains raw accelerometer data from deployments on Ruffs. Each row represents a single accelerometer sample (with one sample every 0.02 seconds). Datetime is derived from the accelerometer clock, so is subject to clock drift and is not synchronised with standard local time. The table includes some data from before and after the logger was on the bird; times when the logger was actually on the bird can be found in logger_deployment_notes.csv (see below). There is a multicolumn index on recording_id and datetime.

| Column       | Description                                |
|--------------|--------------------------------------------|
| recording_id | unique ID code for accelerometer recording |
| datetime     | sample datetime (in fractional seconds)    |
| accx         | x-axis acceleration                        |
| accy         | y-axis acceleration                        |
| accz         | z-axis acceleration                        |

### ruff-acc.db: recordings table

This table is a simple reference for the recording ID codes. Simplified `id` codes (arbitrary numbers between 1 and 30) are used in the acc table to reduce storage requirements, which are here matched to the `recording_id` used across other project files.

| Column       | Description                                                                          |
|-------------------|-----------------------------------------------------|
| id           | unique numeric code representing recording_id in the 'acc' table.                    |
| recording_id | recording ID used across other project files (including logger_deployment_notes.csv) |
| filename     | source file used for database import (not in public dataset)                         |

------------------------------------------------------------------------

# raw-data-other.zip {#raw-data-other.zip}

------------------------------------------------------------------------

### logger_deployment_notes.csv {#logger_deployment_notes.csv}

This file contains contextual information for each accelerometer deployment. Each row represents one deployment, which consists of one logger being turned on in the lab, attached to a bird in the aviary, collected from the bird some days later, then switched off. Each deployment has a unique recording_id that corresponds to a recording_id in the recordings table of ruff-acc.db. Dates and times are synchronised with standard local time. The file includes the following columns:

| Column                            | Description                                               |
|---------------------------------------|---------------------------------|
| recording_id                      | unique identifier for accelerometer deployment            |
| logger_id                         | unique identifier for each accelerometer device           |
| start_date                        | when device was turned on                                 |
| start_time                        | when device was turned on                                 |
| start_dt                          | when device was turned on                                 |
| pre_calibration_start_dt          | when the first calibration was conducted                  |
| post_calibration_start_dt         | when the final calibration was conducted                  |
| stop_date                         | when device was turned off                                |
| stop_time                         | when device was turned off                                |
| stop_dt                           | when device was turned off                                |
| filename                          | csv filename of raw accelerometer recording               |
| capture_date                      | when bird was captured for accelerometer deployment       |
| capture_time                      | when bird was captured for accelerometer deployment       |
| capture_dt                        | when bird was captured for accelerometer deployment       |
| ruff_id_number                    | unique identifier for each bird                           |
| colour_left                       | colour bands on left leg of bird                          |
| colour_right                      | colour bands right leg of bird                            |
| sex                               | sex of bird                                               |
| morph                             | morph of bird                                             |
| ruff_appearance                   | bird description to help identify it during video scoring |
| ruff_nickname                     | short name given to bird                                  |
| release_time                      | when bird was released after accelerometer deployment     |
| release_dt                        | when bird was released after accelerometer deployment     |
| wing_length_mm                    | wing length of bird                                       |
| recapture_disturbance_start_dt    | time of bird recapture at end of the study                |
| cal1_time                         | datetime of the first calibration                         |
| cal1_type                         | type of calibration (desk, flight, or stop)               |
| cal2_time                         | datetime of the second calibration                        |
| cal2_type                         | type of calibration (desk, flight, or stop)               |
| cal3_time                         | datetime of the third calibration                         |
| cal3_type                         | type of calibration (desk, flight, or stop)               |
| cal4_time                         | datetime of the fourth calibration                        |
| cal4_type                         | type of calibration (desk, flight, or stop)               |
| data_analysis_start               | unused column                                             |
| data_analysis_end                 | unused column                                             |
| other_procedures_during_recording | other capture events during study                         |
| other_notes                       | notes                                                     |

### ruff_behaviour_tidy_2022-11-16.csv {#ruff_behaviour_tidy_2022-11-16.csv}

This file includes bird behavioural data, scored from videos using BORIS software. Datetimes are synchronised with standard local time. The file has been tidied and reformatted from the raw BORIS output. It contains the following columns:

| Column              | Description                                           |
|---------------------------------------|---------------------------------|
| beh_event_id        | unique identifier for each behavioural event          |
| recording_id        | unique identifier for each accelerometer deployment   |
| video_analysis_id   | unique identifier for video observation               |
| ruff_id             | unique identifier for bird                            |
| ruff_name           | short name given to bird                              |
| morph               | morph of bird                                         |
| video_file_start_dt | datetime of beginning of video file                   |
| video_start_secs    | latency (secs) between file start and behaviour start |
| video_stop_secs     | latency (secs) between file start and behaviour end   |
| start_dt_real       | start datetime of behaviour                           |
| stop_dt_real        | end datetime of behaviour                             |
| duration_secs       | duration of behaviour                                 |
| behaviour           | behaviour label                                       |

### calibration_recordings_6O_Apr2022.csv {#calibration_recordings_6o_apr2022.csv}

This file gives details of each accelerometer recording used for 6-orientation (6O) calibration. Calibrations were performed during a short recording separate to the main deployment of accelerometers on birds.

| Column                        | Description                                     |
|----------------------------|--------------------------------------------|
| logger_id                     | unique identifier for each accelerometer device |
| model                         | model of accelerometer                          |
| date_fully_charged            | date when accelerometer was charged             |
| firmware                      | firmware version of accelerometer               |
| battery_v_start               | battery voltage at beginning of recording       |
| name_set                      | checklist for setting name of recording         |
| memory_erased                 | checklist for erasing previous data             |
| hz_setting                    | sample rate in Hz                               |
| g_setting                     | max acceleration in g's                         |
| resolution                    | resolution setting of accelerometer             |
| temperature_logging           | temperature logging setting                     |
| start_date                    | start date of accelerometer recording           |
| start_time                    | start time of accelerometer recording           |
| pre_time.calibration_start_dt | start datetime of calibration session           |
| z1_cal_dt                     | start time of Z1 calibration phase              |
| z2_cal_dt                     | start time of Z2 calibration phase              |
| y1_cal_dt                     | start time of Y1 calibration phase              |
| y2_cal_dt                     | start time of Y2 calibration phase              |
| x1_cal_dt                     | start time of X1 calibration phase              |
| x2_cal_dt                     | start time of X2 calibration phase              |
| recording_purpose             | purpose of accelerometer recording              |
| setup_notes                   | notes                                           |
| calibration notes             | notes                                           |
| stop_date                     | end date of recording                           |
| stop_time                     | end time of recording                           |
| stop_dt                       | end datetime of recording                       |
| battery_v_stop                | battery voltage at end of recording             |
| filename                      | filename of recording's csv datafile            |
| general_notes                 | notes                                           |
| other_notes                   | notes                                           |

### 6O_calibration_files {#6o_calibration_files}

This directory contains .csv files with raw accelerometer data recorded during 6-orientation (6O) calibration. Recordings are at 50Hz, and each row represents a single sample (one row per 0.02 second interval). Times are derived from the accelerometer clock, so are subject to clock drift and are not synchronised to standard local time. Additional information about each accelerometer recording is given in `calibration_recordings_6O_Apr2022.csv`. Filenames have the following format:

`[accelerometer model]_[accelerometer ID]_[start date of recording]_[start time of recording]_[sampling rate in hz][max acceleration in gs][resolution setting of accelerometer]_6O_S1, e.g. axy4.5_L24_2022-4-19_171111_50Hz16g10bit_6O_S1`.

Each accelerometer file contains the following columns:

| Column    | Description                                             |
|-----------|---------------------------------------------------------|
| TagID     | unique identifier for each recording (same as filename) |
| Date      | date of accelerometer sample                            |
| Time      | time of accelerometer sample (fractional seconds)       |
| accX      | x-axis acceleration of accelerometer sample             |
| accY      | y-axis acceleration of accelerometer sample             |
| accZ      | z-axis acceleration of accelerometer sample             |
| Temp (?C) | Accelerometer temperature, unused column, usually NA.   |
