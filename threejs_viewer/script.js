const fs = require('fs');
const express = require('express');
const bodyParser = require('body-parser');
const { execSync } = require('child_process');
const {getCurrentWindow, globalShortcut} = require('electron').remote;

//3D Objects
var phone_cam;
var anchor;
var colmap_points;
var arcore_points;
var scene;
var server;
var cameraDisplayOrientedPose;
var camera; //ThreeJS Camera
var controls;
var origin = new THREE.Vector3( 0, 0, 0 );
var red = 0xff0000;
var green = 0x00ff00;
var blue = 0x0000ff;
var yellow = 0xffff00;
var white = 0xffffff;
var orange = 0xffa500;
var pink = 0xFFC0CB;
var useCameraDisplayOrientedPose = true;
var camera_pose;
var local_camera_axes_points;
var x_axis_point;
var y_axis_point;
var z_axis_point;
var cameraWorldCenter;
var cameraWorldCenterPoint;
var debugAnchorPosition;
var debugAnchor;
var arCoreViewMatrix;
var arCoreProjMatrix;
var cameraPoseStringMatrix;

window.onload = function() {

    $(".viewARCoreCame").click(function(){
        //nothing
    });

    //start server
    const app = express();
    app.use(bodyParser.urlencoded({ extended: true, limit: '1mb' }));
    app.use(bodyParser.json({limit: '1mb'}));

    app.post('/', (req, res) => {

        $(".frame").attr('src', 'data:image/png;base64,'+req.body.frameString);

        camera_pose = req.body.cameraPoseLocal.split(','); // switch between cameraPoseLocal and cameraPoseWorld
        var x_local_cam_axis = req.body.x_local_cam_axis.split(',');
        var y_local_cam_axis = req.body.y_local_cam_axis.split(',');
        var z_local_cam_axis = req.body.z_local_cam_axis.split(',');

        //debugAnchorPosition = req.body.debugAnchorPositionForDisplayOrientedPose.split(",");

        var tx = parseFloat(camera_pose[0]);
        var ty = parseFloat(camera_pose[1]);
        var tz = parseFloat(camera_pose[2]);
        var qx = parseFloat(camera_pose[3]);
        var qy = parseFloat(camera_pose[4]);
        var qz = parseFloat(camera_pose[5]);
        var qw = parseFloat(camera_pose[6]);

        phone_cam.position.x = tx;
        phone_cam.position.y = ty;
        phone_cam.position.z = tz;

        cameraWorldCenterPoint.position.x = tx;
        cameraWorldCenterPoint.position.y = ty;
        cameraWorldCenterPoint.position.z = tz;

        var quaternion = new THREE.Quaternion();
        quaternion.fromArray([qx, qy, qz, qw]);
        quaternion.normalize(); // ?
        phone_cam.setRotationFromQuaternion(quaternion);

        x_axis_point.position.x = parseFloat(x_local_cam_axis[0]);
        x_axis_point.position.y = parseFloat(x_local_cam_axis[1]);
        x_axis_point.position.z = parseFloat(x_local_cam_axis[2]);

        y_axis_point.position.x = parseFloat(y_local_cam_axis[0]);
        y_axis_point.position.y = parseFloat(y_local_cam_axis[1]);
        y_axis_point.position.z = parseFloat(y_local_cam_axis[2]);

        z_axis_point.position.x = parseFloat(z_local_cam_axis[0]);
        z_axis_point.position.y = parseFloat(z_local_cam_axis[1]);
        z_axis_point.position.z = parseFloat(z_local_cam_axis[2]);

        var pointsArray = req.body.pointCloud.split("\n");
        pointsArray.pop(); // remove newline

        scene.remove(arcore_points);
        var pointsGeometry = new THREE.Geometry();
        var material =  new THREE.PointsMaterial( { color: green, size: 0.02 } );

        for (var i = 0; i < pointsArray.length; i++) {
            x = parseFloat(pointsArray[i].split(" ")[0]);
            y = parseFloat(pointsArray[i].split(" ")[1]);
            z = parseFloat(pointsArray[i].split(" ")[2]);

            pointsGeometry.vertices.push(
                new THREE.Vector3(x, y, z)
            )
        }
        arcore_points = new THREE.Points( pointsGeometry, material );
        scene.add(arcore_points);

        res.sendStatus(200);
    });

    app.post('/localise', (req, res) => {

        console.log("Localised Action Hit");
        var frameName = req.body.frameName
        console.log(frameName);
        var data_dir = $("#data_to_use").val();
        var query_location = "/Users/alex/Projects/CYENS/colmap_models/"+data_dir+"/"+frameName;
        var pose = req.body.cameraPoseLocal

        fs.writeFileSync(
            query_location,
            req.body.frameString, 'base64', function(err) {
            console.log(err);
            });

        //execSync("sips -r 90 /Users/alex/Projects/CYENS/ar_core_electron_query_images/"+frameName);

        fs.writeFileSync(
            "/Users/alex/Projects/CYENS/colmap_models/"+data_dir+"/query_name.txt",
            frameName,
            function (err) {
                if (err) return console.log(err);
            });

        fs.writeFileSync(
            "/Users/alex/Projects/CYENS/colmap_models/"+data_dir+"/cameraPose.txt",
            pose, function(err) {
            console.log(err);
        });

        console.log("Localizing..")

        execSync("source venv/bin/activate && python3 /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/single_image_localization.py " + data_dir,
            { cwd: '/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/' });

        $(".frame").attr('src', 'data:image/png;base64,'+req.body.frameString);

        camera_pose = pose.split(',');
        var x_local_cam_axis = req.body.x_local_cam_axis.split(',');
        var y_local_cam_axis = req.body.y_local_cam_axis.split(',');
        var z_local_cam_axis = req.body.z_local_cam_axis.split(',');

        var tx = parseFloat(camera_pose[0]);
        var ty = parseFloat(camera_pose[1]);
        var tz = parseFloat(camera_pose[2]);
        var qx = parseFloat(camera_pose[3]);
        var qy = parseFloat(camera_pose[4]);
        var qz = parseFloat(camera_pose[5]);
        var qw = parseFloat(camera_pose[6]);

        phone_cam.position.x = tx;
        phone_cam.position.y = ty;
        phone_cam.position.z = tz;

        cameraWorldCenterPoint.position.x = tx;
        cameraWorldCenterPoint.position.y = ty;
        cameraWorldCenterPoint.position.z = tz;

        var quaternion = new THREE.Quaternion();
        quaternion.fromArray([qx, qy, qz, qw]);
        quaternion.normalize(); // ?
        phone_cam.setRotationFromQuaternion(quaternion);

        x_axis_point.position.x = parseFloat(x_local_cam_axis[0]);
        x_axis_point.position.y = parseFloat(x_local_cam_axis[1]);
        x_axis_point.position.z = parseFloat(x_local_cam_axis[2]);

        y_axis_point.position.x = parseFloat(y_local_cam_axis[0]);
        y_axis_point.position.y = parseFloat(y_local_cam_axis[1]);
        y_axis_point.position.z = parseFloat(y_local_cam_axis[2]);

        z_axis_point.position.x = parseFloat(z_local_cam_axis[0]);
        z_axis_point.position.y = parseFloat(z_local_cam_axis[1]);
        z_axis_point.position.z = parseFloat(z_local_cam_axis[2]);

        var pointsArray = req.body.pointCloud.split("\n");
        pointsArray.pop(); // remove newline

        scene.remove(arcore_points);
        var pointsGeometry = new THREE.Geometry();
        var material =  new THREE.PointsMaterial( { color: green, size: 0.02 } );

        for (var i = 0; i < pointsArray.length; i++) {
            x = parseFloat(pointsArray[i].split(" ")[0]);
            y = parseFloat(pointsArray[i].split(" ")[1]);
            z = parseFloat(pointsArray[i].split(" ")[2]);
            pointsGeometry.vertices.push(
                new THREE.Vector3(x, y, z)
            )
        }

        arcore_points = new THREE.Points( pointsGeometry, material );
        scene.add(arcore_points);

        console.log("Loading 3D points..");

        var points_file_path = "/Users/alex/Projects/CYENS/colmap_models/"+data_dir+"/points3D_AR.txt";
        var quat_file_path = "/Users/alex/Projects/CYENS/colmap_models/"+data_dir+"/quat.txt";
        var trans_file_path = "/Users/alex/Projects/CYENS/colmap_models/"+data_dir+"/trans.txt";
        read3Dpoints(points_file_path);

        renderer.render( scene, camera );
        var colmapPoints = return3Dpoints(points_file_path);

        var quat = fs.readFileSync(quat_file_path).toString();
        var trans = fs.readFileSync(trans_file_path).toString();

        res.status(200).json({ points: colmapPoints, trans: trans, quat: quat });

    });

    app.post('/localise_debug', (req, res) => {

        console.log("Localised Debug Action Hit");
        var pose = req.body.cameraPoseLocal

        $(".frame").attr('src', 'data:image/png;base64,'+req.body.frameString);

        camera_pose = pose.split(',');
        var x_local_cam_axis = req.body.x_local_cam_axis.split(',');
        var y_local_cam_axis = req.body.y_local_cam_axis.split(',');
        var z_local_cam_axis = req.body.z_local_cam_axis.split(',');

        var anchor_position = req.body.anchorPose.split(',');
        console.log(anchor_position);
        anchor.position.x = parseFloat(anchor_position[0]);
        anchor.position.y = parseFloat(anchor_position[1]);
        anchor.position.z = parseFloat(anchor_position[2]);

        var tx = parseFloat(camera_pose[0]);
        var ty = parseFloat(camera_pose[1]);
        var tz = parseFloat(camera_pose[2]);
        var qx = parseFloat(camera_pose[3]);
        var qy = parseFloat(camera_pose[4]);
        var qz = parseFloat(camera_pose[5]);
        var qw = parseFloat(camera_pose[6]);

        phone_cam.position.x = tx;
        phone_cam.position.y = ty;
        phone_cam.position.z = tz;

        cameraWorldCenterPoint.position.x = tx;
        cameraWorldCenterPoint.position.y = ty;
        cameraWorldCenterPoint.position.z = tz;

        var quaternion = new THREE.Quaternion();
        quaternion.fromArray([qx, qy, qz, qw]);
        quaternion.normalize(); // ?
        phone_cam.setRotationFromQuaternion(quaternion);

        x_axis_point.position.x = parseFloat(x_local_cam_axis[0]);
        x_axis_point.position.y = parseFloat(x_local_cam_axis[1]);
        x_axis_point.position.z = parseFloat(x_local_cam_axis[2]);

        y_axis_point.position.x = parseFloat(y_local_cam_axis[0]);
        y_axis_point.position.y = parseFloat(y_local_cam_axis[1]);
        y_axis_point.position.z = parseFloat(y_local_cam_axis[2]);

        z_axis_point.position.x = parseFloat(z_local_cam_axis[0]);
        z_axis_point.position.y = parseFloat(z_local_cam_axis[1]);
        z_axis_point.position.z = parseFloat(z_local_cam_axis[2]);

        var pointsArray = req.body.pointCloud.split("\n");
        pointsArray.pop(); // remove newline

        scene.remove(arcore_points);
        var pointsGeometry = new THREE.Geometry();
        var material =  new THREE.PointsMaterial( { color: green, size: 0.02 } );

        for (var i = 0; i < pointsArray.length; i++) {
            x = parseFloat(pointsArray[i].split(" ")[0]);
            y = parseFloat(pointsArray[i].split(" ")[1]);
            z = parseFloat(pointsArray[i].split(" ")[2]);
            pointsGeometry.vertices.push(
                new THREE.Vector3(x, y, z)
            )
        }

        arcore_points = new THREE.Points( pointsGeometry, material );
        scene.add(arcore_points);

        renderer.render( scene, camera );

        res.status(200);

    });

    server = app.listen(3000, () => console.log(`Started server at http://localhost:3000!`));

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );

    var renderer = new THREE.WebGLRenderer({canvas: document.getElementById( "drawingSurface" )});
    renderer.setSize( window.innerWidth, window.innerHeight );
    document.body.appendChild( renderer.domElement );

    var size = 10;
    var divisions = 10;

    var gridHelper = new THREE.GridHelper( size, divisions );
    scene.add( gridHelper );

    var axesHelper = new THREE.AxesHelper( 5 );
    scene.add( axesHelper );

    var geometry = new THREE.Geometry();
    geometry.vertices.push(
        new THREE.Vector3(1, 1, 0),
        new THREE.Vector3(0.5, 0.5, 0),
        new THREE.Vector3(-1, 1, 0),
        new THREE.Vector3(-0.5, 0.5, 0),
        new THREE.Vector3(-1, -1, 0),
        new THREE.Vector3(-0.5, -0.5, 0),
        new THREE.Vector3(1, -1, 0),
        new THREE.Vector3(0.5, -0.5, 0),
        new THREE.Vector3(0, 0, 1),
        new THREE.Vector3(0, 0, 1.5),
        new THREE.Vector3(0, 0, 2),
        new THREE.Vector3(0, 0, 2.5),
        new THREE.Vector3(0, 0, 3)
    );

    var material =  new THREE.PointsMaterial( { color: white, size: 0.03 } );
    phone_cam = new THREE.Points( geometry, material );
    phone_cam.scale.set(0.1,0.1,0.1);
    scene.add( phone_cam );

    var geometry = new THREE.SphereGeometry( 1, 32, 32 );
    var material = new THREE.MeshPhongMaterial( {color: white } );
    cameraWorldCenterPoint = new THREE.Mesh( geometry, material );
    scene.add( cameraWorldCenterPoint );
    cameraWorldCenterPoint.scale.set(0.015,0.015,0.015);

    var geometry = new THREE.SphereGeometry( 1, 32, 32 );
    var material = new THREE.MeshPhongMaterial( {color: red} );
    x_axis_point = new THREE.Mesh( geometry, material );
    x_axis_point.position.x = 0.1;
    scene.add( x_axis_point );
    x_axis_point.scale.set(0.02,0.02,0.02);

    var geometry = new THREE.SphereGeometry( 1, 32, 32 );
    var material = new THREE.MeshPhongMaterial( {color: green} );
    y_axis_point = new THREE.Mesh( geometry, material );
    y_axis_point.position.y = 0.1;
    scene.add( y_axis_point );
    y_axis_point.scale.set(0.02,0.02,0.02);

    var geometry = new THREE.SphereGeometry( 1, 32, 32 );
    var material = new THREE.MeshPhongMaterial( {color: blue} );
    z_axis_point = new THREE.Mesh( geometry, material );
    z_axis_point.position.z = 0.1;
    scene.add( z_axis_point );
    z_axis_point.scale.set(0.02,0.02,0.02);

    //anchor point
    var geometry = new THREE.SphereGeometry( 1, 32, 32 );
    var material = new THREE.MeshPhongMaterial( {color: yellow} );
    anchor = new THREE.Mesh( geometry, material );
    anchor.position.z = 0;
    scene.add( anchor );
    anchor.scale.set(0.02,0.02,0.02);

    //reference points
    var geometry = new THREE.SphereGeometry( 1, 32, 32 );
    var material = new THREE.MeshPhongMaterial( {color: 0x00FFFF} );
    var reference_point_1 = new THREE.Mesh( geometry, material );
    scene.add( reference_point_1 );
    reference_point_1.scale.set(0.04,0.04,0.04);
    reference_point_1.position.set(-0.5,0,-1);

    var geometry = new THREE.SphereGeometry( 1, 32, 32 );
    var material = new THREE.MeshPhongMaterial( {color: 0xFD00FF} );
    var reference_point_2 = new THREE.Mesh( geometry, material );
    scene.add( reference_point_2 );
    reference_point_2.scale.set(0.04,0.04,0.04);
    reference_point_2.position.set(0.5,0,-1);

    // lights
    var light = new THREE.DirectionalLight( white );
    var ambientLight = new THREE.AmbientLight( pink );
    light.position.set( 50, 50, 50 );
    scene.add( light );
    scene.add(ambientLight);

    controls = new THREE.OrbitControls(camera, renderer.domElement);

    camera.position.set( 0.1, 1, 1 );
    camera.lookAt(scene.position);

    controls.update(); //must be called after any manual changes to the camera's transform

    function animate() {
        requestAnimationFrame( animate );
        // required if controls.enableDamping or controls.autoRotate are set to true
        controls.update();
        renderer.render( scene, camera );
    }

    animate();

    $( ".slider_size" ).slider({
        min: 0.01,
        max: 0.03,
        step: 0.0001,
        slide: function( event, ui ) {
            var size = ui.value;
            colmap_points.material.size = size;
        }
    });

    $( ".slider_scale" ).slider({
        min: 0.5,
        max: 1.5,
        step: 0.005,
        slide: function( event, ui ) {
            var scale = ui.value;
            console.log("Getting 3D points from COLMAP with scale: " + scale);
            execSync('cd /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/ && python3 create_3D_points_for_ARCore_debug.py ' + scale);
            read3Dpoints();
        }
    });
};

function read3Dpoints(file_path){

    scene.remove(colmap_points); // remove previous ones

    var data = fs.readFileSync(file_path);
    data = data.toString().split('\n');

    var geometry = new THREE.Geometry();

    for (var i = 0; i < data.length; i++) {
        xyz = data[i].split(' ');
        x = parseFloat(xyz[0]);
        y = parseFloat(xyz[1]);
        z = parseFloat(xyz[2]);
        geometry.vertices.push(
            new THREE.Vector3(x, y, z)
        )
        r = parseFloat(xyz[4]);
        g = parseFloat(xyz[5]);
        b = parseFloat(xyz[6]);
        geometry.colors.push(
            new THREE.Color("rgb("+r+", "+g+", "+b+")")
        )
    }

    var material =  new THREE.PointsMaterial( { vertexColors: THREE.VertexColors, size: 0.1 } );
    colmap_points = new THREE.Points( geometry, material );

    colmap_points.material.size = 0.015;
    scene.add(colmap_points);
}

function return3Dpoints(file_path){ //same as read3Dpoints but returns them

    var data = fs.readFileSync(file_path);
    data = data.toString().split('\n');

    //local_points.rotation.z = Math.PI/2; // TODO: Do I need this ?

    var points_array = []
    for (var i = 0; i < data.length; i+=1) {
        if(data[i] == ""){
            continue
        }
        xyz_rgb = data[i].split(' ');
        x = parseFloat(xyz_rgb[0]);
        y = parseFloat(xyz_rgb[1]);
        z = parseFloat(xyz_rgb[2]);

        points_array.push(x);
        points_array.push(y);
        points_array.push(z);

        points_array.push(1);

        r = parseFloat(xyz_rgb[4] / 255);
        g = parseFloat(xyz_rgb[5] / 255);
        b = parseFloat(xyz_rgb[6] / 255);

        points_array.push(r);
        points_array.push(g);
        points_array.push(b);
    }

    return points_array
}