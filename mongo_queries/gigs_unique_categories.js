// this shows that "programming" is the only category!
db.gigs
  .distinct('category')
  .forEach(printjson)
