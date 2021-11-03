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

    $(".useCameraDisplayOrientedPose").click(function(){
        useCameraDisplayOrientedPose = true;
    });

    $(".useCameraPose").click(function(){
        useCameraDisplayOrientedPose = false;
    });

    //assign button listeners
    $(".resetCamera").click(function(){
        camera.position.set( 0.1, 1, 1 );
        camera.lookAt(scene.position);
        controls.update();
    });

    $(".exportPoints").click(function(){
        exportARCorePointCloud();
    });

    $(".loadCOLMAPpoints").click(function(){
        get3DPoints();
        read3Dpoints();
        console.log("Done loading points! you can scale them now!");
    });

    $(".viewARCoreCame").click(function(){

        // tx = parseFloat(cameraDisplayOrientedPose[0]);
        // ty = parseFloat(cameraDisplayOrientedPose[1]);
        // tz = parseFloat(cameraDisplayOrientedPose[2]);
        // qx = parseFloat(cameraDisplayOrientedPose[3]);
        // qy = parseFloat(cameraDisplayOrientedPose[4]);
        // qz = parseFloat(cameraDisplayOrientedPose[5]);
        // qw = parseFloat(cameraDisplayOrientedPose[6]);
        //
        // phone_cam_1.position.x = tx;
        // phone_cam_1.position.y = ty;
        // phone_cam_1.position.z = tz ;
        //
        // var rotMatrix = new THREE.Matrix4();
        // var quat = new THREE.Quaternion();
        // quat.fromArray([qx,qy,qz,qw]);
        // quat.normalize();
        // rotMatrix.makeRotationFromQuaternion(quat);
        // rotMatrix.setPosition(tx,ty,tz);
        //
        // var lookAtPoint =  new THREE.Vector4([0, 0, -1, 1]);
        // lookAtPoint.applyMatrix4(rotMatrix);
        //
        // debugger;
        // phone_cam_1.lookAt(0,0,0);

        //controls.update();
    });

    $( ".localiseButton" ).click(function() {
        //localise(cameraDisplayOrientedPose);
    });

    //start server
    const app = express();
    app.use(bodyParser.urlencoded({ extended: true, limit: '1mb' }));
    app.use(bodyParser.json({limit: '1mb'}));

    app.post('/', (req, res) => {

        $(".frame").attr('src', 'data:image/png;base64,'+req.body.frameString);

        if(useCameraDisplayOrientedPose) {
            camera_pose = req.body.cameraDisplayOrientedPose.split(',');
            local_camera_axes_points = req.body.cameraDisplayOrientedPoseLocalAxes.split(",");
            cameraWorldCenter = req.body.cameraDisplayOrientedPoseCamCenter.split(",");
            debugAnchorPosition = req.body.debugAnchorPositionForDisplayOrientedPose.split(",");
            cameraPoseStringMatrix = req.body.cameraDisplayOrientedPoseMatrix;
        }else{
            camera_pose = req.body.cameraPose.split(',');
            local_camera_axes_points = req.body.cameraPoseLocalAxes.split(",");
            cameraWorldCenter = req.body.cameraPoseCamCenter.split(",");
            debugAnchorPosition = req.body.debugAnchorPositionForCameraPose.split(",");
            cameraPoseStringMatrix = req.body.cameraPoseMatrix;
        }

        arCoreViewMatrix = req.body.viewmtx;
        arCoreProjMatrix = req.body.projMatrix;

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

        debugAnchor.position.x = debugAnchorPosition[0];
        debugAnchor.position.y = debugAnchorPosition[1];
        debugAnchor.position.z = debugAnchorPosition[2];

        var quaternion = new THREE.Quaternion();
        quaternion.fromArray([qx, qy, qz, qw]);
        quaternion.normalize(); // ?
        phone_cam.setRotationFromQuaternion(quaternion);

        var x = parseFloat(local_camera_axes_points[0]);
        var y = parseFloat(local_camera_axes_points[1]);
        var z = parseFloat(local_camera_axes_points[2]);
        x_axis_point.position.x = x;
        x_axis_point.position.y = y;
        x_axis_point.position.z = z;

        x = parseFloat(local_camera_axes_points[3]);
        y = parseFloat(local_camera_axes_points[4]);
        z = parseFloat(local_camera_axes_points[5]);
        y_axis_point.position.x = x;
        y_axis_point.position.y = y;
        y_axis_point.position.z = z;

        x = parseFloat(local_camera_axes_points[6]);
        y = parseFloat(local_camera_axes_points[7]);
        z = parseFloat(local_camera_axes_points[8]);
        z_axis_point.position.x = x;
        z_axis_point.position.y = y;
        z_axis_point.position.z = z;

        var anchorPosition = req.body.anchorPosition.split(',');
        var anchor_tx = parseFloat(anchorPosition[0]);
        var anchor_ty = parseFloat(anchorPosition[1]);
        var anchor_tz = parseFloat(anchorPosition[2]);

        anchor.position.x = anchor_tx;
        anchor.position.y = anchor_ty;
        anchor.position.z = anchor_tz;

        var pointsArray = req.body.pointCloud.split("\n");
        pointsArray.pop(); // remove newline

        scene.remove(arcore_points);
        var pointsGeometry = new THREE.Geometry();
        var material =  new THREE.PointsMaterial( { color: blue, size: 0.02 } );

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


        // var pointsArray = req.body.pointCloudByViewMatrix.split("\n");
        // pointsArray.pop(); // remove newline
        //
        // scene.remove(arcore_points_view_matrix);
        // var pointsGeometry = new THREE.Geometry();
        // var material =  new THREE.PointsMaterial( { color: blue, size: 0.02 } );
        //
        // for (var i = 0; i < pointsArray.length; i++) {
        //     x = parseFloat(pointsArray[i].split(" ")[0]);
        //     y = parseFloat(pointsArray[i].split(" ")[1]);
        //     z = parseFloat(pointsArray[i].split(" ")[2]);
        //
        //     pointsGeometry.vertices.push(
        //         new THREE.Vector3(x, y, z)
        //     )
        // }
        // arcore_points_view_matrix = new THREE.Points( pointsGeometry, material );
        // scene.add(arcore_points_view_matrix);

        res.sendStatus(200);
    });

    app.post('/localise', (req, res) => {

        console.log("Localised Action Hit");
        // debugger;
        var frameName = req.body.frameName
        console.log(frameName);
        var query_location = "/Users/alex/Projects/CYENS/ar_core_electron_query_images/"+frameName;
        var pose = req.body.cameraDisplayOrientedPose

        fs.writeFileSync(
            query_location,
            req.body.frameString, 'base64', function(err) {
            console.log(err);
            });

        //execSync("sips -r 90 /Users/alex/Projects/CYENS/ar_core_electron_query_images/"+frameName);

        fs.writeFileSync(
            "/Users/alex/Projects/CYENS/ar_core_electron_query_images/query_name.txt",
            frameName,
            function (err) {
                if (err) return console.log(err);
            });

        fs.writeFileSync(
            "/Users/alex/Projects/CYENS/ar_core_electron_query_images/cameraPose.txt",
            pose, function(err) {
            console.log(err);
        });

        console.log("Localizing..")
        execSync("source venv/bin/activate && python3 /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/single_image_localization.py " + frameName,
            { cwd: '/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/' });

        console.log("Loading 3D points..")
        read3Dpoints();
        renderer.render( scene, camera );

        var colmapPoints = return3Dpoints();
        var finalPose = returnServerPose();

        res.status(200).json({ points: colmapPoints, serverPose: finalPose });

    });

    app.post('/getModel', (req, res) => {
        // var colmapPoints = getModel();
        var colmapPoints = loadPoints3DFromFile();
        res.status(200).json({ points: colmapPoints });
    });


    app.post('/reload', (req, res) => {
        getCurrentWindow().reload();
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
    var material = new THREE.MeshPhongMaterial( {color: yellow} );
    anchor = new THREE.Mesh( geometry, material );
    scene.add( anchor );
    anchor.scale.set(0.03,0.03,0.03);

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

    var geometry = new THREE.SphereGeometry( 1, 32, 32 );
    var material = new THREE.MeshPhongMaterial( {color: orange} );
    debugAnchor = new THREE.Mesh( geometry, material );
    scene.add( debugAnchor );
    debugAnchor.scale.set(0.02,0.02,0.02);

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

function get3DPoints(){
    console.log("Getting 3D points from COLMAP");
    execSync('cd /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/ && python3 create_3D_points_for_ARCore_debug.py');
}

function read3Dpoints(){

    scene.remove(colmap_points); // remove previous ones

    const file_path = '/Users/alex/Projects/CYENS/colmap_models/points3D_AR.txt';

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

    //colmap_points.rotation.z = Math.PI/2;
    scene.add(colmap_points);
}

function return3Dpoints(){ //same as read3Dpoints but returns them

    const file_path = '/Users/alex/Projects/CYENS/colmap_models/points3D_AR.txt';

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

function returnServerPose(){

    const file_path = '/Users/alex/Projects/CYENS/colmap_models/final_pose.txt';

    var data = fs.readFileSync(file_path);
    data = data.toString().split('\n');
    data.pop()
    return data
}

function getModel(){
    const file_path = '/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/all_points3D.txt';

    var data = fs.readFileSync(file_path);
    data = data.toString().split('\n');

    return data;
}

function loadPoints3DFromFile(){
    const file_path = '/Users/alex/Projects/EngDLocalProjects/LEGO/fullpipeline/colmap_data/data/points3D_AR.txt';

    var data = fs.readFileSync(file_path);
    data = data.toString().split('\n');

    return data;
}

function exportMatrixString(matrix, name){
    fs.writeFileSync("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/"+name+".txt",
        matrix, 'utf8', function(err) {
            console.log(err);
        });
}

function exportARCorePointCloud() {

    var arcore_points_String = "";
    for (var i = 0; i < arcore_points.geometry.vertices.length; i++) {

        var x = arcore_points.geometry.vertices[i].x;
        var y = arcore_points.geometry.vertices[i].y;
        var z = arcore_points.geometry.vertices[i].z;

        arcore_points_String += x + " " + y + " " + z + " " + 1 +"\n"
    }

    fs.writeFileSync("/Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/colmap_data/data/arcore_pointCloud.txt",
        arcore_points_String, 'utf8', function(err) {
            console.log(err);
        });
}

function debug_COLMAP_points(scale){
    execSync('cd /Users/alex/Projects/EngDLocalProjects/Lego/fullpipeline/ && python3 create_3D_points_for_ARCore_debug.py ' + scale);
    read3Dpoints();
}
