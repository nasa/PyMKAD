# PyMKAD

The world-wide aviation system is one of the most complex dynamical systems ever developed and is generating data at an extremely rapid rate. Most modern commercial aircraft record several hundred flight parameters including information from the guidance, navigation, and control systems, the avionics and propulsion systems, and the pilot inputs into the aircraft. These parameters may be continuous measurements or binary/categorical measurements recorded in one second intervals for the duration of the flight. Currently, most approaches to aviation safety are reactive, meaning that they are designed to react to an aviation safety incident or accident. PyMKAD is a novel approach based on the theory of multiple kernel learning to detect potential safety anomalies in very large data bases of discrete and continuous data from world-wide operations of commercial fleets. This code address an anomaly detection problem which includes both discrete and continuous data streams, where we assume that the discrete streams influence on the continuous streams. We also assume that atypical sequence of events in the discrete streams can lead to off-nominal system performance.  

The objective of this project is to automate the analysis of flight safety incidents in a way that combines both analysis of discrete and continuous parameters. 

This repository contains the following files in its top level directory:

* [PythonCode](PythonCode)  
The source code of the repository includes: preprocessing modules, the main mkad code, and a post processing visualization tool. The code is uses a command line interface and a json file for configuring. 

* [documentation](documentation)  
Documents describing how to configure and run the program, as well as how to interpret the results. 


* [MKAD NOSA 2019.pdf](MKAD%20NOSA%202019.pdf)  
Licensing for MKAD




## Contact Info

NASA Point of contact: Nikunj Oza <nikunj.c.oza@nasa.gov>, Data Science Group Lead.

For questions regarding the research and development of the algorithm, please contact Bryan Matthews <bryan.l.matthews@nasa.gov>, Senior Research Engineer.


## Copyright and Notices

Notices:

Copyright © 2019 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

Disclaimers

No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.

