using System.Text;
using Newtonsoft.Json;

public class ChatStuff
{
    private static HttpClient Http = new HttpClient();
  
    public async Task GenerateResponse()
    {
        var chatGPT = new ChatGPT();
        Console.WriteLine("Waiting for response...");
        string response = await chatGPT.GenerateLoremIpsum();
        Console.WriteLine(response);
    }
}

// This code defines a ChatGPT class that makes an API call to the OpenAI API to generate some lorem ipsum text.
// It uses the HttpClient class to make a POST request to the OpenAI API, passing in some JSON data that includes the prompt, model, and max_tokens.
// The response is read as a string and then deserialized into a dynamic object, and the text is returned.

public class ChatGPT {

  private static HttpClient Http = new HttpClient();
  public async Task<string> GenerateLoremIpsum()
  {
      // Replace [INSERT_YOUR_OWN_API_KEY] with a valid OpenAI API key
      var apiKey = "[your api key here]";
      Http.DefaultRequestHeaders.Add("Authorization", $"Bearer {apiKey}");

      // JSON content for the API call
      var jsonContent = new
      {
          prompt = "Give me some lorem ipsum text",
          model = "gpt-3.5-turbo",
          max_tokens = 1000
      };

      // Make the API call
      var responseContent = await Http.PostAsync("https://api.openai.com/v1/completions", new StringContent(JsonConvert.SerializeObject(jsonContent), Encoding.UTF8, "application/json"));

      // Read the response as a string
      var resContext = await responseContent.Content.ReadAsStringAsync();

      // Deserialize the response into a dynamic object
      var data = JsonConvert.DeserializeObject<dynamic>(resContext);
      return data.choices[0].text.ToString();
  }
}