<?php
$conn=new mysqli('localhost','root','','bechmobile');
$sql="select * from sell";
$result= mysqli_query($conn,$sql);
?>

<!DOCTYPE html>
<html>
<head>
	<meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">
	<title>Buy from here</title>
	<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/css/bootstrap.min.css" integrity="sha384-9aIt2nRpC12Uk9gS9baDl411NQApFmC26EwAOH8WgZl5MYYxFfc+NcPb1dKGj7Sk" crossorigin="anonymous">

  <link rel="stylesheet" type="text/css" href="../css/responsive-bootstrap.css">
</head>

<body style="background: url(../img/mobile_header.jpg) no-repeat; background-size:cover;">
<header>
  <nav  id ='header-nav'>
  <nav class="navbar navbar-expand-lg navbar-dark bg-dark" style="background-color: transparent;" >
  <a class="navbar-brand" href="#" style="color: white;">Bechmobile</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNavDropdown" aria-controls="navbarNavDropdown" aria-expanded="false" aria-label="Toggle navigation" style="float:right;">
    <span class="navbar-toggler-icon" style="color: white;"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNavDropdown">
    <ul class="navbar-nav">
      <li class="nav-item ">
        <a class="nav-link" href="home.html">Home </a>
      <li class="nav-item">
        <a class="nav-link" href="sell.html">Sell</a>
      </li>
        <div class="dropdown-menu dropdown-dark bg-dark" aria-labelledby="navbarDropdownMenuLink">
          <a class="dropdown-item" href="templates/predictprice.html" style="color: grey;">Predict Price</a>
        </div>
      </li>
      <li class="nav-item dropdown active">
        <a class="nav-link dropdown-toggle" href="#" id="navbarDropdownMenuLink" role="button" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">Buy Phone<span class="sr-only">(current)</span></a>
        <div class="dropdown-menu dropdown-dark bg-dark" aria-labelledby="navbarDropdownMenuLink">
          <a class="dropdown-item" href="#" style="color: grey;">New</a>
          <a class="dropdown-item" href="retrieve.php" style="color: grey;">Used</a>
        </div>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="#">Wishlist</a>
      </li>
    </ul>
    <div class="dropdown show ml-auto">
  <a class=" nav-link navbar-dark  btn btn-secondary dropdown-toggle" href="#" role="button" id="dropdownMenuLink" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">
    Profile
  </a>

  <div class="dropdown-menu bg-dark" aria-labelledby="dropdownMenuLink">
    <a class="dropdown-item" href="#" style="color:grey;">Log Out</a>
  </div>
</nav>
</nav>
  </header>

<section>
	<?php
		while($row=mysqli_fetch_array($result))
		{
	?>
		<div style="background: linear-gradient(white,grey); display:inline-block; padding:2%; margin:4%;" class="col-sm-2 col-md-3 col-lg-4">
      <div>
        <?php echo '<img src="data:image;base64,'.base64_encode($row['mobile_picture']).'"alt="Image" style="width:100%">';?>
      </div>
			<?php echo $row['model_name'],"<br>",$row['selling_prize'],"<br>",$row['contact_number'],"<br>"; ?>
		</div>
	<?php
		}
	?>
</section>


<script src="https://code.jquery.com/jquery-3.5.1.slim.min.js" integrity="sha384-DfXdz2htPH0lsSSs5nCTpuj/zy4C+OGpamoFVy38MVBnE+IbbVYUew+OrCXaRkfj" crossorigin="anonymous"></script>
<script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" crossorigin="anonymous"></script>
<script src="https://stackpath.bootstrapcdn.com/bootstrap/4.5.0/js/bootstrap.min.js" integrity="sha384-OgVRvuATP1z7JjHLkuOU7Xw704+h835Lr+6QL9UvYjZE3Ipu6Tp75j7Bh/kR0JKI" crossorigin="anonymous"></script>

<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jquery/2.1.3/jquery.min.js"></script>
<script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/jqueryui/1.11.2/jquery-ui.min.js"></script>
<script type="text/javascript" src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/js/bootstrap.min.js"></script>
</body>
</html>