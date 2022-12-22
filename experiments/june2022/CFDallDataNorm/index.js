



async function runExample() {
    

  // Create an ONNX inference session with default backend.
  const session = new onnx.InferenceSession();
 
  await session.loadModel("./CFDallNORM.onnx");

  var x = new Float32Array(1, 10);

  //x = [   38.0000,     0.0000,   300.0000,    27.0000, 308750.3750,    12.1540, 359906.3125,  1459.8170, 24494.0000, 87618.0156]
  

  var x = [];
  x[0] = document.getElementById('box0').value;
  x[1] = document.getElementById('box1').value;
  x[2] = document.getElementById('box2').value;
  x[3] = document.getElementById('box3').value;
  x[4] = document.getElementById('box4').value;
  x[5] = document.getElementById('box5').value;
  x[6] = document.getElementById('box6').value;
  x[7] = document.getElementById('box7').value;
  x[8] = document.getElementById('box8').value;
  x[9] = document.getElementById('box9').value;

  

  const tensorX = new onnx.Tensor(x, 'float32', [1, 10]);
  
   
  
  const outputMap = await session.run([tensorX]);
  const outputData = outputMap.get('output1');
  
  
 

  // PREDS DIV 
  const predictions = document.getElementById('predictions');
  
 


  predictions.innerHTML = `<hr> Got an output Tensor of size ${outputData.data.length} with values being: <br/> <table>
  <tr><td> o_tuyere_exit_velo_m_s</td><td> ${outputData.data[0]} </td></tr>
  <tr><td> o_tuyere_t_k</td>          <td> ${outputData.data[1]} </td></tr> 
  <tr><td> o_raceway_flame_temp_k </td><td> ${outputData.data[2]} </td></tr>
  <tr><td>o_raceway_coal_burn_perce </td><td>${outputData.data[3]} </td></tr> <tr> <td> o_raceway_volume_m </td><td> ${outputData.data[4]} </td></tr> <tr> <td> o_raceway_depth _m </td><td> ${outputData.data[5]} </td></tr> <tr><td> o_shaft_co_utiliz </td><td> ${outputData.data[6]} </td></tr>   <tr><td> o_shaft_h2_utiliz </td><td> ${outputData.data[7]} </td></tr>   <tr><td> o_shaft_top_gas_temp_c </td><td> ${outputData.data[8]}</td></tr> <tr><td> o_shaft_press_drop_pa </td><td> ${outputData.data[9]} </td></tr> <tr><td>o_shaft_coke_rate_kg_thm</td><td>${outputData.data[10]}</td></tr> <tr><td>o_shaft_cohesive_zone_tip_height_m</td>
      <td>${outputData.data[11]} </td></tr>
      <tr><td>o_shaft_cohes_zone_root_height_m </td>
      <td> ${outputData.data[12]} </td></tr>
      <tr> <td> o_shaft_co_v_perc </td><td> ${outputData.data[13]} </td></tr> <tr><td>o_shaft_co2_v_perc</td><td> ${outputData.data[14]} </td></tr> <tr><td> o_shaft_h2_v_perce</td><td> ${outputData.data[15]}</td></tr> <tr><td> o_shaft_n2_v_perc </td><td> ${outputData.data[16]} </td></tr> </table>   `;
 



}
