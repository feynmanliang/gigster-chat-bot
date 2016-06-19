function numGigsWithCalculators() {
  pipeline = [
    { $match: { 'calculator': { $exists: true } } },
    { $group: { _id: 1, count: { $sum: 1 } } }
  ];

  R = db.runCommand({
    'aggregate': 'gigs',
    'pipeline': pipeline
  });

  printjson(R)
}

// this shows that there are 2443 gigs with calculators
// numGigsWithCalculators()

function sampleCalculators() {
  pipeline = [
    { $match: { 'calculator': { $exists: true } } },
    { $limit: 5 }
  ];

  R = db.runCommand({
    'aggregate': 'gigs',
    'pipeline': pipeline
  });

  printjson(R)
}

sampleCalculators()
