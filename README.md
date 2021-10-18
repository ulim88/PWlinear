# PWlinear
We newly proposed a piecewise linear labeling method for enhancing the speed adaptability in human gait phase estimation.

The details of the proposed methodology can be found in the manuscript, "Effect of torso kinematics and piecewise linear label on gait phase estimation for enhancing speed adaptability."


This repository contains the following contents: (1) dataset, and (2) TNSRE_Gait_estimation.py

(1) dataset

    This contains the processed data of 50 subjects in 5 different walking speeds: C1-C5.
    Each csv file is named as 'PWCombined_20XXXXX_YY_ZZ.csv.'
    20XXXXX refers to the subject id, and YY refers to the walking speed, while ZZ refers to the number of trials.
    In each dataset, 11 columns are contained.
        - Index: data index
        - LeftHS: linear label based on left heel-strike
        - RightHS: linear label based on right heel-strike
        - LThighAngle: calculated left thigh angular position
        - RThighAngle: calculated right thigh angular position
        - TorsoAngle: calculated torso angular position
        - LThighVelocity: calculated left thigh angular velocity
        - RThighVelocity: calculated right thigh angular velocity
        - TorsoVelocity: calculated torso angular velocity
        - PieceWiseLeft: piecewise linear label based on left heel-strike
        - PieceWiseRight: piecewise linear label based on right heel-strike

(2) TNSRE_Gait_estimation.py

    This is the main code for training the given data using the proposed networks.
