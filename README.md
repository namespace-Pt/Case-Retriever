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
- [x] knn search
#### Highlighting
- [x] highlight matching by bolding
#### Facets
- [x] term facets
### Backend
- django
### Frontend
- bootstrap


1. 裁定书很少包含案情