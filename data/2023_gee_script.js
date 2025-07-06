// STEP 1: Define ROI
var roi = ee.Geometry.Rectangle([76.8, 18.5, 77.2, 18.9]); // Change if needed

// STEP 2: Load Sentinel-2 SR
var s2 = ee.ImageCollection("COPERNICUS/S2_SR")
  .filterBounds(roi)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10));

function maskCloudsAndAddIndices(image) {
  var scl = image.select('SCL');
  var mask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10));
  image = image.updateMask(mask);

  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI').toFloat();
  var evi = image.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))',
    {
      NIR: image.select('B8'),
      RED: image.select('B4'),
      BLUE: image.select('B2')
    }
  ).rename('EVI').toFloat();
  
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI').toFloat();
  var msi = image.select('B11').divide(image.select('B8')).rename('MSI').toFloat();
  var savi = image.expression(
    '((NIR - RED) / (NIR + RED + 0.5)) * 1.5',
    {
      NIR: image.select('B8'),
      RED: image.select('B4')
    }
  ).rename('SAVI').toFloat();

  return image.addBands([ndvi, evi, ndwi, msi, savi]);
}

s2 = s2.map(maskCloudsAndAddIndices);

var months = ee.List.sequence(1, 12);

months.getInfo().forEach(function(m) {
  var start = ee.Date.fromYMD(2023, m, 1);
  var end = start.advance(1, 'month');
  var monthName = start.format('MMM').getInfo().toLowerCase();

  var indices = s2.filterDate(start, end)
                  .mean()
                  .select(['NDVI', 'EVI', 'NDWI', 'MSI', 'SAVI']);

  Export.image.toDrive({
    image: indices,
    description: 'Indices_' + monthName + '_2023',
    folder: 'GEE_MultiIndex_2023',
    fileNamePrefix: 'multi_' + monthName + '_2023',
    region: roi,
    scale: 10,
    maxPixels: 1e13
  });
});
