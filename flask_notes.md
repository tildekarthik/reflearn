
# These are my course notes for later reference - Not my original work
## The course is : Udemy : Python and Flask Bootcamp by Jose Portilla - He is the author
### I use this as my cut-paste reference sheet - kindof snippets :-)


# HTML Forms
`<form action= method=>`
- actions - points to where the submit should be sent - web address - not used in course
- Method - Get or Post

```
<input> has type = email, text, password, submit, radio - bound to same name
<select> selects an option
<label for ='id' same as in input 'id'>
<text area>
```

Example:
``` html
<form method="get">
          <h3>Do you already own a dog?</h3>
          <label for="yes">Yes</label>
          <input type="radio" id="yes" name="dog_choice" value="yes">
          <label for="no">No:</label>
          <input type="radio" id="no" name= "dog_choice" value="no">
          <p>How clean is your house (Rated 1-3 with 3 being cleanest))</p>
          <select name="stars">
            <option value="Great">3</option>
            <option value="Okay">2</option>
            <option value="Bad">1</option>
          </select>
          <p>Any other comments?</p>
          <textarea name="name" rows="8" cols="80"></textarea>
          <input type="submit" name="" value="Submit Feedback">
        </form>
```

# Bootstrap
- Put the following lines inside `<head>` after `<\title>` - copy from bootstrap page

``` html
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/css/bootstrap.min.css" integrity="sha384-WskhaSGFgHYWDcbwN70/dfYBj47jz9qbsMId/iRN3ewGhXQFZCSftd1LZCfmhktB" crossorigin="anonymous">
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js" integrity="sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo" crossorigin="anonymous"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.3/umd/popper.min.js" integrity="sha384-ZMP7rVo3mIykV+2+9J3UJ46jBk0WLaUAdn689aCwoqbBJiSnjAK/l8WvCWPIPm49" crossorigin="anonymous"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.1.1/js/bootstrap.min.js" integrity="sha384-smHYKdLADwkXOn1EmN1qk/HfnUcbVRZyYmZ4qpPea6sjB/pTJ0euyQp0Mk8ck+5T" crossorigin="anonymous"></script>
```
- jumbotron is an interesting div class
- nav bar can be used in all pages by drawing upon the base.html and extending the same

# Flask basic commands

```
from flask import Flask
from flask import request
app = Flask(__name__)
@app.route('/')
def index():
    # Grab the visitors User Agent information
    browser_info = request.headers.get('User-Agent')
    return '<h2>Here is your browser info:</h2> <p>{}</p>'.format(browser_info)
```
- Use headers.dict()o know all the details of the headers
- Use code below to read and process the content from details in address bar


```
@app.route('/puppy_latin/<name>')
def puppylatin(name):
    # Puppy Latin the name that comes in!
    pupname = ''
    if name[-1]=='y':
        pupname = name[:-1]+'iful'
    else:
        pupname = name+'y'
    return "<h1>Hi {}! Your puppylatin name is {} </h1>".format(name,pupname)

```
- a trick later explained to use as integer the id if used in a query is like this: (also note the or 404 trick)

```
@blog_posts.route('/<int:blog_post_id>')
def blog_post(blog_post_id):
	blogpost = BlogPost.query.get_or_404(blog_post_id)
```


- Using requests with only action and no methods to process data

``` html
<form action="{{url_for('thank_you')}}">
       <label for="first">First Name</label>
        <input type="text" name="first">
        <label for="last">Last Name</label>
        <input type="text" name="last">
        <input type="submit" value="Submit Form">
</form>
```

and thankyou route use: 

```
	first = request.args.get('first')
    last = request.args.get('last')
    return render_template('06-thankyou.html',first=first,last=last)`
```

- A fancy error handler?

```
@app.errorhandler(404)
def page_not_found(e):
    return render_template('06-404.html'), 404
```


# Jinja templates and control flow
- `{{var}}` is used to show the content of the variable
- control flow for and if else - `{% %}`

```
<ul>
{% for pup in puppies %}
      <li>{{pup}}</li>
{% endfor %}
</ul>
    <p>We can also add if/else statements</p>
{% if 'Rufus' in puppies %}
      <p>Found you in the list of puppies Rufus!</p>
{% else %}
      <p>Hmm, Rufus isn't in this list.</p>
{% endif %}
```

# Main applications - app.py
- Imports and general

```
from flask import Flask, render_template, session, redirect, url_for, session
app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'
```

- Feed the encapsuated form and do something on submit (also session is a global object available across the apps)

Typically on a valid submit something is retrieved from db or something is written to the database. Here were using session object:

```
@app.route('/', methods=['GET', 'POST'])
def index():
    # Create instance of the form.
    form = InfoForm()
    if form.validate_on_submit():
        # Grab the data from the breed on the form.
        session['breed'] = form.breed.data
        session['neutered'] = form.neutered.data
        session['mood'] = form.mood.data
        session['food'] = form.food_choice.data
        session['feedback'] = form.feedback.data
        return redirect(url_for("thankyou"))
    return render_template('01-home.html', form=form)
```

see below for sessions object used in thankyou, though not passed to it
```
<li>Breed: {{session['breed']}}</li>
  <li>Neutered: {{session['neutered']}}</li>
  <li>Mood: {{session['mood']}}</li>
  <li>Food: {{session['food']}}</li>
  <li>Feedback: {{session['feedback']}}</li>
```

- Flash messages without any html

```
if form.validate_on_submit():
        flash("You just clicked the button!")
```

- Trick to show flashed messages nicely in the template html file

```
{# get_flashed_messages() is auto sent to the template with the flash() call #}
      {% for mess in get_flashed_messages()  %}
      <div class="alert alert-warning alert-dismissible fade show" role="alert">
        <button type="button" class="close" data-dismiss="alert" aria-label="Close" class="fade close">
          <span aria-hidden="true">&times;</span>
        </button>
        {{mess}}
        </div>
      {% endfor %}
```




# Flask forms 
## - class definition - forms.py```
from flask_wtf import FlaskForm
from wtforms import (StringField, BooleanField, DateTimeField,
                     RadioField,SelectField,TextField,
                     TextAreaField,SubmitField)
from wtforms.validators import DataRequired

class AddForm(FlaskForm):
    pup_id = IntegerField("Id of Puppy: ")
    breed = StringField('What breed are you?',validators=[DataRequired()])
    neutered  = BooleanField("Have you been neutered?")
    mood = RadioField('Please choose your mood:', choices=[('mood_one','Happy'),('mood_two','Excited')])
    food_choice = SelectField(u'Pick Your Favorite Food:',
                          choices=[('chi', 'Chicken'), ('bf', 'Beef'),
                                   ('fish', 'Fish')])
    feedback = TextAreaField()
    submit = SubmitField('Submit')
```

#### Html - Dont forget the hidden tag

``` html
<form method="POST">
	
    {# This hidden_tag is a CSRF security feature. #}
    {{ form.hidden_tag() }}
    {{ form.breed.label }} {{ form.breed() }}
    {{ form.submit() }}
</form>
```

# Flask Models (Classes used in data bases)
- Initialization

```
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
# This grabs our directory
basedir = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__)
# Connects our Flask App to our Database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
Migrate(app,db)
```


- Class files - each is a database table
Puppy has one owner and many toys - 3 tables created below with relationship 

```
class Puppy(db.Model):
    __tablename__ = 'puppies'
    id = db.Column(db.Integer,primary_key = True)
    name = db.Column(db.Text)
    # This is a one-to-many relationship
    # A puppy can have many toys
    toys = db.relationship('Toy',backref='puppy',lazy='dynamic')
    # This is a one-to-one relationship
    # A puppy only has one owner, thus uselist is False.
    # Strong assumption of 1 dog per 1 owner and vice versa.
    owner = db.relationship('Owner',backref='puppy',uselist=False)
    def __init__(self,name):
        # Note how a puppy only needs to be initalized with a name!
        self.name = name
    def __repr__(self):
        if self.owner:
            return f"Puppy name is {self.name} and owner is {self.owner.name}"
        else:
            return f"Puppy name is {self.name} and has no owner assigned yet."
    def report_toys(self):
        print("Here are my toys!")
        for toy in self.toys:
            print(toy.item_name)
```
        
```
class Toy(db.Model):
    __tablename__ = 'toys'
    id = db.Column(db.Integer,primary_key = True)
    item_name = db.Column(db.Text)
    # Connect the toy to the puppy that owns it.
    # We use puppies.id because __tablename__='puppies'
    puppy_id = db.Column(db.Integer,db.ForeignKey('puppies.id'))
    def __init__(self,item_name,puppy_id):
        self.item_name = item_name
        self.puppy_id = puppy_id
```

```
class Owner(db.Model):
    __tablename__ = 'owners'
    id = db.Column(db.Integer,primary_key= True)
    name = db.Column(db.Text)
    # We use puppies.id because __tablename__='puppies'
    puppy_id = db.Column(db.Integer,db.ForeignKey('puppies.id'))
    def __init__(self,name,puppy_id):
        self.name = name
        self.puppy_id = puppy_id
```

## CRUD commands
### Create
```
my_puppy = Puppy('Rufus',5)
db.session.add(my_puppy)
db.session.commit()
```

### Read
- Note lots of ORM filter options here.
- filter(), filter_by(), limit(), order_by(), group_by()
- Also lots of executor options
- all(), first(), get(), count(), paginate()

`all_puppies = Puppy.query.all() # list of all puppies in table print(all_puppies)`

-  Grab by id
```
puppy_one = Puppy.query.get(1)
print(puppy_one)
print(puppy_one.age)
```

- Filters

```
puppy_sam = Puppy.query.filter_by(name='Sammy') # Returns list
print(puppy_sam)
```

### Update
- Grab your data, then modify it, then save the changes.

```
first_puppy = Puppy.query.get(1)
first_puppy.age = 10
db.session.add(first_puppy)
db.session.commit()
```

### Delete

```
second_pup = Puppy.query.get(2)
db.session.delete(second_pup)
db.session.commit()
```

## Full crud views in main app 

- most apps are some form of these 4 operations - create , read, update and delete

```
@app.route('/')
def index():
    return render_template('home.html')
@app.route('/add', methods=['GET', 'POST'])
def add_pup():
    form = AddForm()
    if form.validate_on_submit():
        name = form.name.data
        # Add new Puppy to database
        new_pup = Puppy(name)
        db.session.add(new_pup)
        db.session.commit()
        return redirect(url_for('list_pup'))
    return render_template('add.html',form=form)
@app.route('/list')
def list_pup():
    # Grab a list of puppies from database.
    puppies = Puppy.query.all()
    return render_template('list.html', puppies=puppies)
@app.route('/delete', methods=['GET', 'POST'])
def del_pup():
    form = DelForm()
    if form.validate_on_submit():
        id = form.id.data
        pup = Puppy.query.get(id)
        db.session.delete(pup)
        db.session.commit()
        return redirect(url_for('list_pup'))
    return render_template('delete.html',form=form)`
```


# Large application structuring
- Link to text file for referennce - See text file to understand the tree structure
- This is app.py, this is the main file called.

```
from myproject import app
from flask import render_template
@app.route('/')
def index():
    return render_template('home.html')
if __name__ == '__main__':
    app.run(debug=True)
```
	
- `__init.py__` file update

```
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
app = Flask(__name__)
# Often people will also separate these into a separate config.py file 
app.config['SECRET_KEY'] = 'mysecretkey'
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'data.sqlite')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
Migrate(app,db)
# NOTE! These imports need to come after you've defined db, otherwise you will
# get errors in your models.py files.
## Grab the blueprints from the other views.py files for each "app"
from myproject.puppies.views import puppies_blueprint
from myproject.owners.views import owners_blueprint
app.register_blueprint(owners_blueprint,url_prefix="/owners")
app.register_blueprint(puppies_blueprint,url_prefix='/puppies')
```

- Note: There is an alternate approach in the large app where the url_prefix is not given as an argument. Check which works

- `models.py` file update

```
from myproject import db
```

...followed by all the classes...

- forms.py [In each folder underneath Puppy and Owner]

```
from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, SubmitField
```

...followed by form class definitions...

- views.py[In each folder underneath - Puppy separate owner separate]

```
from flask import Blueprint,render_template,redirect,url_for
from myproject import db
from myproject.models import Owner
from myproject.owners.forms import AddForm
owners_blueprint = Blueprint('owners',
                              __name__,
                              template_folder='templates/owners')
@owners_blueprint.route('/add', methods=['GET', 'POST'])
def add():
    form = AddForm()
    if form.validate_on_submit():
        name = form.name.data
        pup_id = form.pup_id.data
        # Add new owner to database
        new_owner = Owner(name,pup_id)
        db.session.add(new_owner)
        db.session.commit()
        return redirect(url_for('puppies.list'))
    return render_template('add_owner.html',form=form)
```


# Authentication
## Main app and views
- Note there is no logout page - once logged out sent to home - only a flash message

``` python
from myproject import app,db
from flask import render_template, redirect, request, url_for, flash,abort
from flask_login import login_user,login_required,logout_user
from myproject.models import User
from myproject.forms import LoginForm, RegistrationForm
from werkzeug.security import generate_password_hash, check_password_hash


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/welcome')
@login_required
def welcome_user():
    return render_template('welcome_user.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You logged out!')
    return redirect(url_for('home'))


@app.route('/login', methods=['GET', 'POST'])
def login():

    form = LoginForm()
    if form.validate_on_submit():
        # Grab the user from our User Models table
        user = User.query.filter_by(email=form.email.data).first()
        
        # Check that the user was supplied and the password is right
        # The verify_password method comes from the User object
        # https://stackoverflow.com/questions/2209755/python-operation-vs-is-not

        if user.check_password(form.password.data) and user is not None:
            #Log in the user

            login_user(user)
            flash('Logged in successfully.')

            # If a user was trying to visit a page that requires a login
            # flask saves that URL as 'next'.
            next = request.args.get('next')

            # So let's now check if that next exists, otherwise we'll go to
            # the welcome page.
            if next == None or not next[0]=='/':
                next = url_for('welcome_user')

            return redirect(next)
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()

    if form.validate_on_submit():
        user = User(email=form.email.data,
                    username=form.username.data,
                    password=form.password.data)

        db.session.add(user)
        db.session.commit()
        flash('Thanks for registering! Now you can login!')
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)`
```

## Add the following to the `__init__.py`

``` python
from flask_login import LoginManager
# Create a login manager object
login_manager = LoginManager()
```

After you do the usual `app=Flask(__name__)` , database etc upto Migrate, insert

``` python
# We can now pass in our app to the login manager
login_manager.init_app(app)
# Tell users what view to go to when they need to login.
login_manager.login_view = "login"  # In large app - users.login as a separate folder
```
 
## Add the following to models.py

```
from myproject import db,login_manager
from werkzeug.security import generate_password_hash,check_password_hash
from flask_login import UserMixin
# By inheriting the UserMixin we get access to a lot of built-in attributes
# which we will be able to call in our views!
# is_authenticated()
# is_active()
# is_anonymous()
# get_id()
# The user_loader decorator allows flask-login to load the current user
# and grab their id.
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)

class User(db.Model, UserMixin):
    # Create a table in the db
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key = True)
    email = db.Column(db.String(64), unique=True, index=True)
    username = db.Column(db.String(64), unique=True, index=True)
    password_hash = db.Column(db.String(128))
    def __init__(self, email, username, password):
        self.email = email
        self.username = username
        self.password_hash = generate_password_hash(password)
    def check_password(self,password):
        # https://stackoverflow.com/questions/23432478/flask-generate-password-hash-not-constant-output
        return check_password_hash(self.password_hash,password)`
```

## Add to Forms.py
```
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, EqualTo
from wtforms import ValidationError
from flask_wtf.file import FileField,FileAllowed

from flask_login import current_user
from puppycompanyblog.models import User

class LoginForm(FlaskForm):
    email = StringField('Email',validators=[DataRequired(),Email()])
    password = PasswordField('Password',validators=[DataRequired()])
    submit = SubmitField('Log In')

class RegistrationForm(FlaskForm):
    email = StringField('Email',validators=[DataRequired(),Email()])
    username = StringField('UserName',validators=[DataRequired()])
    password = PasswordField('Password',validators=[DataRequired(),EqualTo('pFloginass_confirm',message='Passwords must match!')])
    pass_confirm = PasswordField('Confirm Password',validators=[DataRequired()])
    submit = SubmitField('Register!')

    def check_email(self,field):
        if User.query.filter_by(email=field.data).first():
            raise ValidationError('Your email has been registered already!')

    def check_username(self,field):
        if User.query.filter_by(username=field.data).first():
            raise ValidationError('Your username has been registered already!')


class UpdateUserForm(FlaskForm):

    email = StringField('Email',validators=[DataRequired(),Email()])
    username = StringField('UserName',validators=[DataRequired()])
    picture = FileField('Update Profile Picture',validators=[FileAllowed(['jpg','png'])])
    submit = SubmitField('Update')

    def check_email(self,field):
        if User.query.filter_by(email=field.data).first():
            raise ValidationError('Your email has been registered already!')

    def check_username(self,field):
        if User.query.filter_by(username=field.data).first():
            raise ValidationError('Your username has been registered already!')

```
## Add to html for register

```
{% extends "base.html" %}
{% block content %}
<form method="POST">
    {# This hidden_tag is a CSRF security feature. #}
    {{ form.hidden_tag() }}
    {{ form.email.label }} {{ form.email() }}<br>
    {{ form.username.label }} {{ form.username() }}<br>
    {{ form.password.label }} {{ form.password() }}<br>
    {{ form.pass_confirm.label }} {{ form.pass_confirm() }}<br>
    {{ form.submit() }}
</form>
{% endblock %}

```
- Fancy use of username and is authenticated in base.html

```
{% if current_user.is_authenticated %}
    <p>Hi {{ current_user.username }}!</p>
{% else %}
    <p>Please login or register!</p>
{% endif %}`
```

# Tips and tricks for configuration

a. Static folder expose to internet
`app = Flask(__name__, static_url_path='/static')`
b. Debug mode
c. Host IP
d. Deployment
e. Secret key
f. Changing schema in database and migrations - command prompt





# Flask database migration steps at the OS prompt

```
set (windows) or export(linux) FLASK_APP=models.py or app.py)
flask db init` - Maybe done only when there is a schema change
flask db migrate -m "Some blah commit style message"
flask db upgrade
```
