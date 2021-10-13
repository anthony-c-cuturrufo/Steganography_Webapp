from pyngrok import ngrok


public_url = ngrok.connect(port='80')
print(public_url)
