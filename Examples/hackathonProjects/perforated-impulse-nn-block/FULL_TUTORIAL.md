# Full Tutorial

This set of instructions is for folks who have never used Edge Impulse before.  The walkthrough sets you up with a tutorial project doing keyword spotting from audio files, then shows you how to add the perforated block to that project.

## Edge Impulse Setup and Project Creation

- Start by creating an account.  Go to [Edge Impulse](https://www.edgeimpulse.com){:target="_blank"} and click on Login at the top right.  At the bottom is the option to "Sign Up".  Click that button and go through the steps to get your account set up.
  - This will load you into a tutorial.  Feel free to go through or just click "get started manually" to follow this classificaiton-only tutorial.
- Once your account is created load the [Bird Sound Classifier](https://studio.edgeimpulse.com/public/16060/latest){:target="_blank"}.  This is the public project with the most views on the platform. It processes audio files and aims to classify them into three classes: House Sparrow, Rose Ringed Parakeet, or random noise that is neither.
- On the top right click "clone this project. 
  - Select a name such as "Tutorial: Responding to your voice-perforated"
  - Optional - Choose to make it a public project so it is easier to share
  - Note: The clone project function can take a few minutes, this is normal
- Once the clone completes you will automatically be loaded into the cloned project
- In the cloned project click on "Create Impulse" on the left.  This will show you an overview of the project and the blocks involved.
  - Time series data is the block which handles and contains your raw data
  - Audio (MFCC) is a block which extracts features from audio signals using Mel Frequency Cepstral Coefficients.  Just know this means converting audio into a format that is easier to process.
  - Neural Network (Keras) is the learning block which trains a neural network on the dataset
  - Output Features is not a block, but just specifies that for this Impulse there are the three output classes described above
- Click on "NN Classifier" on the left.
  - You can see in the center the various configuration settings. Make notes about the following:
    - Learning rate is 0.0005
    - If you click "Advanced Training Settings" the batch size is 32
    - Also make note of the full neural network configuraiton which had layers in the following order
      - Reshape - 13 Columns
      - 2d conv - 8 filters, 3 kernel width, 1 layer
      - Dropout - rate 0.25
      - 2d conv - 16 filters, 3 kernel width, 1 layer
      - Dropout - rate 0.25
      - Fully connected output layer 3 classes
  - Once you have reviewed the settings click "Save and Train" at the bottom
    - Scroll back up and you can see a terminal pop up on the top right.  This will show your status and detauls as your neural network trains for 100 epochs.  Process will also take a few minutes to process.
- Once training completes make a note of your accuracy score.  When we ran this we acheived a score of 90.4%.  This means if you deployed this model to the edge it would make mistakes approximately 9.6% of the times it is triggered.

Now that you have trained your baseline, proceed to the next step to try the same dataset with the same architecture, while perforating your model.

## Adding the New Block

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/PerforatedAI/PerforatedAI?devcontainer_path=Examples%2FhackathonProjects%2Fexample-custom-ml-block-pytorch%2F.devcontainer%2Fdevcontainer.json){:target="_blank"}

 - This requires signing in, or creating a Github account if you do not have one
 - Use default settings and click "Create codespace"
 - At the bottom center of the window that loads is the terminal to run commands
 - In the teriminal run the following to install the edge-ipulse-cli, login to your account, and upload the block.  You may have to type these by hand if copy-paste is not working.  Copy-paste also may require you to use "ctrl+shift+p" and then click the "paste" option that comes up.
 - The `edge-impulse-blocks init` command will require you to login with the account you just created.  If you logged in with a google account, or similar, you will have to create a password.
   - To create a password go back to your edge impulse window and click your porfile in the top right.
   - Then click "Account Settings" then "Password".
   - Click "Set a Password" and create a password for yourself.
 - `edge-impulse-blocks push` is also a command that can take up to 10 minutes to run.
 - After the commands fully run navigate to your profile's [custom blocks](https://studio.edgeimpulse.com/studio/profile/custom-blocks){:target="_blank"} and make sure the new block is present.
```
    git checkout nn_customize 
    npm install -g edge-impulse-cli --silent
    cd Examples/hackathonProjects/perforated-impulse-nn-block/
    edge-impulse-blocks init
    edge-impulse-blocks push
```

Now that your block is uploaded you can do the following:
 - Open your project which has a classifier block
 - Click "Create Impulse" on the left
 - Under your classifier block, sometimes called Neural Network (Keras),click "add learning block"
 - Select "Add" next to the name of the block you just uploaded, likely "Perforated Classification"
 - Click "Save Impulse" on the right.
 - Open your classificaiton block and the new perforated block in two tabs in your browser
   - These are on the left, likely labeled "NN Classifier" and "Perforated Classificaiton"
 - Recreate your exact settings in the new perforated block
   - This usually means to check learning rate and batch size as well as network definition
   - For network definition for the Perforated block you will have to select each layer type and type with text the settings for the layer
     - This is performed by selecting layers in the configuration settings labeled "Layer _ Type" and then details in the new window that comes up.
     - These are comma-separated values and the text will show up in order to replace with numbers
     - If a layer that is in your classifier block does not exist in the perforated block it is done automatically
     - The final fully connected layer also is done automatically, no need to edit the text in the "settings" window for that block.
     - The exact settings for the tutorial block you need to do are described in the top section in the bullet described as "configuration settings"
 - We also reccomend selecting `GPU` for training processor, this will run much faster.
 - Click "Save & train" to run the new block!
 - Check your results
   - In the text window you can scroll up and see the exact scores before and after dendrites were added
   - Check if performance improved
   - Check if this performance actually beat your original nn classifier performance
     - When we ran this dataset with default settings we show 
     - This reduces the error rate from the original 9.6% down to 6.4%.  An error reduction of 33% just by making this switch!
 - If you want, feel free to try other hyperparameter settings
   - Often a smaller model with dendrites will outperform a larger nn classifier
   - If you want a higher accuracy turn up the number of dendrites
   - Check the post training analysis in terminal, if your pre-dendrite score is lower than your original score try setting `improvement-threshold` to low.
 - Please let us know how it goes for you!