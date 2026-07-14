PROGRAMMING FOR AI — FINAL ASSIGNMENT
Course: AI 2000 
Institute: IIT Hyderabad
Student Name: Desaboina Sri Sathwik
Roll Number: AI24BTECH11007
Submission Type: ZIP Submission

----------------------------------------------------------------------
OVERVIEW
----------------------------------------------------------------------

This assignment implements a mini machine learning framework from scratch, similar in structure to PyTorch, and applies it to multiple datasets.

It covers:
- Automatic differentiation
- Neural network layers
- Optimization algorithms
- Model evaluation on Fashion-MNIST and Spambase datasets

The project is modular and organized under the folder "my_ml_lib".
Each experiment (Q3–Q5) is implemented as a separate Jupyter notebook in the "experiments" directory.

----------------------------------------------------------------------
FOLDER STRUCTURE
----------------------------------------------------------------------

FINAL_BOILERPLATE/
│
├── data/
│   └── spambase.data
│
├── experiments/
│   ├── Q3_SpamClassification.ipynb
│   ├── Q4_AutogradVisualization.ipynb
│   ├── Q5_CapstoneShowdown.ipynb
│
├── my_ml_lib/
│   ├── autograd/          - Autograd engine (Value class, gradients)
│   ├── datasets/          - Dataset loading utilities
│   ├── linear_models/     - Linear and logistic regression
│   ├── model_selection/   - Data split and evaluation functions
│   ├── naive_bayes/       - Naive Bayes model
│   └── nn/                - Neural network components
│       ├── modules/       - Layers, activations, containers
│       ├── base.py
│       └── optim.py
├── Report.pdf             - Gives the report on Q3 and Q4 outputs
├── create_best_model.py   - Trains and saves the best model
├── visualize.py           - Visualization of losses and accuracy
└── saved_models/          - Trained model checkpoints

----------------------------------------------------------------------
HOW TO RUN
----------------------------------------------------------------------

1. Extract the FINAL_BOILERPLATE.zip file.
2. Open any Jupyter notebook from the "experiments" folder, such as:
   - Q3_SpamClassification.ipynb
   - Q4_AutogradVisualization.ipynb
   - Q5_CapstoneShowdown.ipynb
3. Run all cells sequentially to execute the code for each part.

Note: Ensure the "my_ml_lib" folder is in the same directory as your notebooks before running.

----------------------------------------------------------------------
KEY FILES AND THEIR ROLES
----------------------------------------------------------------------

my_ml_lib/autograd/       - Core autograd engine implementing Value class.
my_ml_lib/nn/modules/     - Linear, Sequential, and activation layers.
my_ml_lib/nn/optim.py     - Implements SGD optimizer.
my_ml_lib/model_selection/- Data splitting and evaluation utilities.
visualize.py              - Plots training and validation performance.
saved_models/             - Stores trained model weights and checkpoints.

----------------------------------------------------------------------
DATASETS USED
----------------------------------------------------------------------

Fashion-MNIST  - 10-class grayscale image classification (28x28 images)
Spambase       - Binary text classification (spam detection)

All dataset files are available in the "data" folder.

----------------------------------------------------------------------
EXPERIMENTS SUMMARY
----------------------------------------------------------------------

Q3_SpamClassification.ipynb   - Spam classification using Naive Bayes and Logistic Regression.
Q4_AutogradVisualization.ipynb - Gradient flow visualization using the custom autograd engine.
 

----------------------------------------------------------------------
OUTPUT
----------------------------------------------------------------------

- Trained model files are saved in the "saved_models" folder.
- Training and accuracy plots are shown in notebooks or can be generated using "visualize.py".

----------------------------------------------------------------------
NOTES
----------------------------------------------------------------------

- Python version: 3.10 or higher
- Dependencies: numpy, pandas, matplotlib, tqdm
- No external ML frameworks (like PyTorch or TensorFlow) were used.

----------------------------------------------------------------------
AUTHOR
----------------------------------------------------------------------

Name: Desaboina Sri Sathwik
Roll Number: AI24BTECH11007
Course: AI 2000
Institute: IIT Hyderabad
