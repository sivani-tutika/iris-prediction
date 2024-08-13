# predictor/views.py
from django.shortcuts import render
from .forms import PredictionForm
import joblib

def predict(request):
    prediction = None
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            model = joblib.load('predictor/model.pkl')
            data = [
                form.cleaned_data['sepal_length'],
                form.cleaned_data['sepal_width'],
                form.cleaned_data['petal_length'],
                form.cleaned_data['petal_width']
            ]
            prediction = model.predict([data])[0]
    else:
        form = PredictionForm()

    return render(request, 'predictor/predict.html', {'form': form, 'prediction': prediction})
