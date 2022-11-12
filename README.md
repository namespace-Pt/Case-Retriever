## Environments
- `elasticsearch==8.4.1`
- `django==4.0.3`
- `bootstrap==5.1.3`

## Instruction
```
bin/elasticsearch-plugin install analysis-smartcn
bin/elasticsearch

cd app
python manager.py runserver
```

## Features
### Searching
- [x] search in chinese
- [x] search multiple fields by one query
- [x] search exact match phrases by quoting the query
- [x] knn search
- [x] term facets
- [x] highlight matching by bolding
- [x] processing icon
- [x] searching similar cases button
- [x] explain

### Backend
- django
### Frontend
- bootstrap
