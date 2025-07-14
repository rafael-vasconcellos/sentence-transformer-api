import requests


response = requests.post("http://localhost:8000/api/similarity", json={ 
    "texts1": ["Eu gosto de andar de bicicleta nas manhãs de domingo.", "A entrega está programada para amanhã à tarde."],
    "texts2": ["Aos domingos de manhã, eu adoro pedalar.", "A remessa vai chegar amanhã no período da tarde."],
})

print(response.json())