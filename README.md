# Home Automation Machine Learning Project

Introduction
This project is a machine learning application designed to generate home automation sequences. \
It uses a combination of data preparation, machine learning, and a RESTful API to process, model, \
and serve automation data.



## Development
The project is developed in Python, using the FastAPI framework for the API, and Keras for the \ 
machine learning component. SQLite is used for the database and stored locally in the project directory. \
\
Create new venv
```
python -m venv .venv
```

Activate venv (powershell)
```
.venv/Scripts/Activate.ps1
```

Exit venv by executing
```
deactivate
```
### Dependencies
Install all dependencies
```
$ pip install -r requirements.txt
```

Update  requirements.txt 
```
pip3 freeze > requirements.txt 
```

## Building docker image
```
$ docker build -t automation-api .
```
Run docker image
```
$ docker run -d -p 8000:8000 automation-api
```


## Running server
Run the server
```
$ python -m uvicorn main:app
```

Run local server, but accessible over network
```
$ python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## Testing

All tests are located in the `src` directory.\
A test database is also included in this folder and is used for testing.

To test all the existing tests, run the following command in the root directory of the project
```
$ python -m unittest discover -s . -p "*_test.py"
```

## Linting
Linting is important to reduce errors and improve the overall quality of your code.

To lint the project
```
$ pylint *.py 
```

## Database
The project uses SQLite for the database. The database file is `database.db`, and the schema is defined in `database_handler.py`.

To interact with the database, use the `DatabaseHandler` class, which provides methods for saving and retrieving data.
### Automation table
```sql
CREATE TABLE "automations" (
	"id"	INTEGER,
	"value"	TEXT NOT NULL,
	"type"	INTEGER NOT NULL,
	"createdOn"	TEXT,
	"updatedOn"	TEXT,
	"deleted"	INTEGER DEFAULT 0,
	PRIMARY KEY("id" AUTOINCREMENT)
);
```

##  Conclusion
This project is a great example of how machine learning can be used in the field of home automation. By training a model on existing automation data, we can generate new automation sequences, potentially uncovering useful patterns and saving time for users.
