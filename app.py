from flask import Flask, request, render_template, send_from_directory, jsonify
import pickle
import numpy as np

app = Flask(__name__)

with open('model2.pkl', 'rb') as file:
    model = pickle.load(file)

# Define mappings from prediction integers to messages
prediction_messages = {
    1: "Business: This target type is mainly characterized by a variety of commercial activities or commercial premises for profit such as various entertainment venues and various private business premises. Common examples are bars, shops, etc.",
    2: "Government (general): This target type mainly includes politicians, former politicians, and political venues. Terrorist attacks are generally targeted at politicians, e.g., in response to their election campaigns or political measures taken.",
    3: "Police: This target category includes all police personnel and their offices and residences, such as police stations, prisons, checkpoints, etc. It also includes the vehicles used by the police, such as police cars.",
    4: "Military: This target category includes military bases, military personnel, and military equipment such as tanks, armored vehicles, and aircraft.",
    5: "Abortion-related: This target category includes healthcare settings where abortion-related activities take place, targeting customers, security guards, etc.",
    6: "Airport & aircraft: This target type mainly includes civil aviation aircraft and airports, excluding fighter aircraft.",
    7: "Government (diplomacy): This target category includes infrastructure related to the diplomatic field such as embassies, consulates, and all personnel within these facilities.",
    8: "Educational institution: This target type includes educational infrastructure such as school buildings, websites, buses, and personnel including teachers and students.",
    9: "Food or water supply: This target type includes premises and workers involved in water or food-related facilities such as water plants and food manufacturing plants.",
    10: "Journalists & media: This target category includes individuals and facilities associated with the news media industry, such as journalists, photographers, and their premises.",
    11: "Maritime: This target category includes various maritime facilities such as ships, cruise ships, yachts, and fishing boats.",
    12: "NGO (Non-governmental organizations): This target category includes facilities and staff of non-state institutions.",
    13: "Other: This target type encompasses attacks that do not fit into the other specified categories.",
    14: "Private citizens & property: This target type includes attacks on private citizens in public places, such as robbery, hijacking, and kidnapping incidents.",
    15: "Religious figures/institutions: This target type includes religious institutions such as temples, churches, shrines, and personnel such as priests.",
    16: "Telecommunication: This target type includes facilities of various communications equipment such as signal towers, telephones, broadcasting stations, etc.",
    17: "Terrorist/non-state militias: This target type refers specifically to attacks by various terrorist groups or their members.",
    18: "Tourists: This target type includes tourism-related facilities and services, such as tourists, cruise ships, sightseeing buses, amusement parks, and attractions.",
    19: "Transportation: This target type includes public transportation equipment such as cars, subways, high-speed trains, and transport channels such as roads, bridges, and tunnels.",
    20: "Unknown: This target type refers to incidents where the target is not precisely known or specified.",
    21: "Utilities: This target type includes national power stations, oil and gas development sites, electrical equipment, and other energy-related facilities.",
    22: "Violent political parties: This target type refers to political parties that use non-peaceful means and violence to achieve their goals."
}

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # Get the input features from the form
        input_features = [int(x) for x in request.form.values()]
        final_features = np.array(input_features).reshape(1, -1)
        print(input_features)
        print(final_features)

        # Perform prediction
        prediction = model.predict(final_features)
        print("prediction", prediction)
        
        # Extract the predicted class index (assuming classes are 0-indexed)
        predicted_class_index = prediction[0]
        print("class_index", predicted_class_index)
        
        output_message = prediction_messages.get(predicted_class_index, "Default message if prediction not mapped")
        
        # Return the prediction as a JSON response
        return jsonify({'prediction': output_message})
    return render_template('services.html')

@app.route('/services')
def services():
    return render_template('services.html')

if __name__ == '__main__':
    app.run(debug=True)
