from keras.models import load_model
from flask import Flask, render_template, request
from wtforms import Form, TextField, validators, SubmitField, DecimalField, IntegerField, TextAreaField, SelectField
from wtforms.validators import Length
from helpers import generate_from_seed, amiModel, articleModel

model_ami, word_index_ami = amiModel()
model_article, word_index_article = articleModel()

# Create app
app = Flask(__name__)


class ReusableForm(Form):
    """User entry form for entering specifics for generation"""
    # Starting seed
    seed = TextAreaField("Enter your text:", validators=[
                     validators.InputRequired(), Length(max=1000000)])
    
    # Diversity of predictions
    diversity = DecimalField('Enter threshold:', default=0.5,
                             validators=[validators.InputRequired(),
                                         validators.NumberRange(min=0.5, max=0.99,
                                                                message='Threshold must be between 0.5 and 0.99')])
    textType = SelectField('Type of text:', choices=[('Article', 'Article'), ('Meeting', 'Meeting')], validators=[validators.InputRequired()])
 # Submit button
    submit = SubmitField("Submit text")

# Home page
@app.route("/", methods=['GET', 'POST'])
def home():
    """Home page of app with form"""
    # Create form
    form = ReusableForm(request.form)

    # On form entry and all conditions met
    if request.method == 'POST' and form.validate():
        # Extract information
        seed = request.form['seed']
        threshold = request.form['diversity']
        textType = str(request.form['textType'])
        
        if textType == "Meeting":
            model = model_ami
            word_index = word_index_ami
        
        if textType == "Article":
            model = model_article
            word_index = word_index_article
        
        print("success!")
        
        return render_template('seeded.html', input=generate_from_seed(model=model, word_index = word_index, seed=seed, threshold=threshold))
    # Send template information to index.html
    return render_template('index.html', form=form)

    # Run app
    app.run(host="0.0.0.0", port=80)
