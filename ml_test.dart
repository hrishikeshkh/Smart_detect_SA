import 'package:ml_algo/ml_algo.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';
Future<void> main() async {
  List<double> initial_coef = [1.0, 10.0];
  //append the integer a at the end of our data listA
  //convert the intial coef to a vector
  final vec = Vector.fromList(initial_coef);
  // Do something with the data
  //train function goes here
  final initial_vec = Matrix.fromColumns([vec]);
  final features = DataFrame([
    ['feature_1', 'feature_2', 'output'],
    [2, 2, 12],
    [3, 3, 18],
    [4, 4, 24],
    [5, 5, 30],
  ]);
  final model = LinearRegressor.SGD(features, 'output', fitIntercept: false, initialCoefficients: initial_vec);

  print('Coefficients: ${model.coefficients}');
}