#version 330 core

in vec3 FragPos; // Fragment position in world space
in vec3 Normal;  // Normal vector in world space

out vec4 FragColor; // Final fragment color

// Light properties
uniform vec3 lightPos;  // Light position in world space
uniform vec3 lightColor; // Light color
uniform vec3 viewPos;   // Camera position in world space

// Material properties
uniform vec3 ambientColor;   // Ambient color
uniform vec3 diffuseColor;   // Diffuse color
uniform vec3 specularColor;  // Specular color
uniform float shininess;     // Shininess factor

void main()
{
    // Ambient lighting
    vec3 ambient = ambientColor * lightColor;

    // Diffuse lighting
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * diffuseColor * lightColor;

    // Specular lighting (Blinn-Phong)
    vec3 viewDir = normalize(viewPos - FragPos);
    vec3 halfwayDir = normalize(lightDir + viewDir);
    float spec = pow(max(dot(norm, halfwayDir), 0.0), shininess);
    vec3 specular = spec * specularColor * lightColor;

    // Combine results
    vec3 result = ambient + diffuse + specular;
    FragColor = vec4(result, 1.0); // Set the fragment color
}
