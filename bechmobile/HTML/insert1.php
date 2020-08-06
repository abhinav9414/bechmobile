<?php
 $mobile_picture =addslashes(file_get_contents($_FILES["mobile_picture"]["tmp_name"]));
 $model_name =$_POST['model_name'];
 $selling_prize =$_POST['selling_prize'];
 $contact_number =$_POST['contact_number'];
 //echo  $model_name;
 $conn=new mysqli('localhost','root','','bechmobile');
 
 
 
 
 $sql="insert into sell(id,model_name,selling_prize,contact_number,mobile_picture) values('','$model_name','$selling_prize','$contact_number','$mobile_picture')";
 $run= mysqli_query($conn,$sql);
 if($run==TRUE){
 	header('Location: sell.html');
 	echo "inserted correctly please wait for your reponse";
 }
 else {
 	echo "please put appropro";
 }
 ?>
 
 