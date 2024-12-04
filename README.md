# AI-Spam-Detector
 Here I have coded a spam detecting AI which combines a trained BERT model and GPT-3.5-Turbo to either chat with the user or determine if a message is spam all through a UI.

 I wrote this software purley as a challenge to try and train a model rather then just use an imported model. It proved to be difficult, but I successfully trained a BERT model on over 5000 data points, and then integrated that into a chat function with OpenAI.

[Software Demo Video](https://youtu.be/QKp6x1zwyAU)

## Development Enviroment
I used a python virtual enviroment to house my packages so any user with an openAI api key can use this project without additional downloads. 
* To activate the virtual enviroment, inside the AI-SPAM-CHECKER directory run `source ./myenv/bin/activate `

I also used the C# blazor framework for my frontend
* To run this, navigate to the BlazorApp folder and run `dotnet run`

I also used python's fastAPI for the backend to handel the logic between BERT and OpenAI. 
* To run this enviroment, naivgate to backend and run `uvicorn app:app --reload`

## Useful Websites

* [Stack Overflow](https://stackoverflow.com/questions/75774873/openai-api-error-this-is-a-chat-model-and-not-supported-in-the-v1-completions)
  
## Future Work
* make a functional app that can be integrated into local softwares to help prevent seniors from being scammed
