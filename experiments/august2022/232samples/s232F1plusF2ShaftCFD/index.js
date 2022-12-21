



async function runExample() {
    

  // Create an ONNX inference session with default backend.
  const session = new onnx.InferenceSession();
 
  await session.loadModel("./s232F1plusF2_CFDshaftSIO.onnx");

  var x = new Float32Array(1, 6);

 
  

  var x = [];
  x[0] = document.getElementById('box0').value;
  x[1] = document.getElementById('box1').value;
  x[2] = document.getElementById('box2').value;
  x[3] = document.getElementById('box3').value;
  x[4] = document.getElementById('box4').value;
  x[5] = document.getElementById('box5').value;
 

  
  const tensorX = new onnx.Tensor(x, 'float32', [1, 6]);
  
   
  
  const outputMap = await session.run([tensorX]);
  const outputData = outputMap.get('output1');
  
  
 

  // PREDS DIV 
  const predictions = document.getElementById('predictions');
  
 


  predictions.innerHTML = `<hr> Got an output Tensor of size ${outputData.data.length} with values being: <br/> 
 <table>
  <tr><td> o_shaft_co_utiliz</td><td> ${outputData.data[0]} </td></tr>
  <tr><td> o_shaft_h2_utiliz</td><td> ${outputData.data[1]} </td></tr> 
  <tr><td> o_shaft_top_gas_temp_c</td><td> ${outputData.data[2]} </td></tr>
  <tr><td> o_shaft_press_drop_pa</td><td> ${outputData.data[3]} </td></tr> 
  <tr><td> o_shaft_coke_rate_kg_thm</td><td> ${outputData.data[4]} </td></tr> 
 </table>   `;
 



}
