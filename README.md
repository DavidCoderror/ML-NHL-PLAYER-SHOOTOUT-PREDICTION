Nov 16th 2025,

At the Start of this project, I had ran into some wildly big issues. In my data, there already existed shootout data, which at he time thought it would be very uselful.
It was not. It was the exact opposite. It caused overfitting. I spent days on it trying to think of a solution.
I had to get the ideas of what kind of predictions its making. So I decided to make a feature and list of the top players to see how good it is.
With S% being the output, it would cause the players who have no shootout attempts to have a garanteed less than 2% chances at scoring.
Even Connor McDavid had that chance! Which is very bad of a model. So essentially I had to swithc things up.
I tried using other features as possible targets/outputs. Which honestly made the model horrendously inaccurate.
I eventually came aorund hte idea of instead focusing on shootouts... what about makes them up?
So essentially its the idea of scoring game winners under pressure. The idea of "Clutching". 
At the moment General + Stress + Experimentales features are in in the outputs.
So I decided to engineer some features togheter, slwoly but surely it was looking better. Still at the moment it needs some work.
I have been trying other types of Models to see how effective they are. Main focus still being on RandomForest seems to be the best Model at hand. 
Might try more models.

-DC
