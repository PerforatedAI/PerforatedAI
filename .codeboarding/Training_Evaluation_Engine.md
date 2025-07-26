```mermaid
graph LR
    PerforatedAI_Training["PerforatedAI.Training"]
    PerforatedAI_Evaluation["PerforatedAI.Evaluation"]
    PerforatedAI_Data["PerforatedAI.Data"]
    PerforatedAI_Models["PerforatedAI.Models"]
    PerforatedAI_Metrics["PerforatedAI.Metrics"]
    PerforatedAI_Config["PerforatedAI.Config"]
    PerforatedAI_Utils["PerforatedAI.Utils"]
    PerforatedAI_API["PerforatedAI.API"]
    PerforatedAI_Training -- "consumes data from" --> PerforatedAI_Data
    PerforatedAI_Training -- "trains models defined in" --> PerforatedAI_Models
    PerforatedAI_Evaluation -- "uses to compute scores" --> PerforatedAI_Metrics
    PerforatedAI_Evaluation -- "utilizes for support tasks" --> PerforatedAI_Utils
    PerforatedAI_Training -- "reports progress to" --> PerforatedAI_Metrics
    PerforatedAI_Training -- "retrieves hyperparameters from" --> PerforatedAI_Config
    PerforatedAI_Training -- "utilizes for support tasks" --> PerforatedAI_Utils
    PerforatedAI_API -- "offers high-level training functions to" --> PerforatedAI_Training
```

[![CodeBoarding](https://img.shields.io/badge/Generated%20by-CodeBoarding-9cf?style=flat-square)](https://github.com/CodeBoarding/GeneratedOnBoardings)[![Demo](https://img.shields.io/badge/Try%20our-Demo-blue?style=flat-square)](https://www.codeboarding.org/demo)[![Contact](https://img.shields.io/badge/Contact%20us%20-%20contact@codeboarding.org-lightgrey?style=flat-square)](mailto:contact@codeboarding.org)

## Details

One paragraph explaining the functionality which is represented by this graph. What the main flow is and what is its purpose.

### PerforatedAI.Training
Manages optimizers, loss functions, and the core training loops. It is responsible for iterating through data, performing forward and backward passes, and updating model parameters.


**Related Classes/Methods**: _None_

### PerforatedAI.Evaluation
Provides tools and routines for calculating performance metrics, validating models, and generating reports based on trained models and test datasets.


**Related Classes/Methods**: _None_

### PerforatedAI.Data
Handles data loading, preprocessing, augmentation, and batching, providing structured input to the training and evaluation processes.


**Related Classes/Methods**: _None_

### PerforatedAI.Models
Defines and manages neural network architectures, including pre-built models, custom model definitions, and model serialization/deserialization.


**Related Classes/Methods**:

- `perforatedai.pb_models`


### PerforatedAI.Metrics
Provides a collection of performance metrics (e.g., accuracy, loss, precision, recall) and mechanisms for tracking and reporting them during training and evaluation.


**Related Classes/Methods**: _None_

### PerforatedAI.Config
Manages configuration parameters, hyperparameters, and experiment settings, allowing for reproducible and flexible experimentation.


**Related Classes/Methods**: _None_

### PerforatedAI.Utils
Contains common utility functions, helper methods, and general-purpose tools used across various components of the library, promoting code reusability.


**Related Classes/Methods**:

- `perforatedai.pb_utils`


### PerforatedAI.API
Exposes high-level, user-friendly interfaces and functions for initiating training runs, evaluating models, and interacting with the core functionalities of the library.


**Related Classes/Methods**: _None_



### [FAQ](https://github.com/CodeBoarding/GeneratedOnBoardings/tree/main?tab=readme-ov-file#faq)